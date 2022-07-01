import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rube.data.tokenize import index_words

CENTURY = 5200


class RawDataCleaner:
    def __init__(self, raw):
        self.raw_data = raw
        self.data = None
        self.n_samples = None
        self.stock_vocab = None
        self.n_periods = None
        self.n_users = None
        self.max_seen_q = None

    def transform_data(self,
                       max_accepted_quantity=32,
                       stock_vocab_size=2000,
                       user_vocab_size=2000,
                       truncate=True,
                       period_in_weeks=None,
                       min_visits=0,
                       min_baskets=0,
                       min_average_spend=0):
        """
        :param raw: (dataframe) raw data set
        :param max_accepted_quantity: (int) if truncate is false remove all entries with more than max_accepted_quantity
                                            else truncate the quantity of all entries with > maq to maq
        :param stock_vocab_size: (int) maximum number of products to encode
        :param user_vocab_size: (int) maximum number of users to encode
        :param truncate: (bool) see max_accepted_quantity
        :param period_in_weeks: we have the option of breaking the dataset into periods each of length period_in_weeks.
                                Setting a default here of 5200 to ensure that the default will be one period.
        :param min_visits: minimum number of times a customer must show up in the data set to be included in cleaned data
        :param min_baskets: minimum number of baskets a product must show up in in order to be included in the cleaned data
        :param min_average_spend: minimum average spend of a customer to be included in the cleaned data
        """
        self.data, self.n_samples, self.stock_vocab, self.n_periods, self.n_users, self.max_seen_q = \
            transform_data(self.raw_data, stock_vocab_size, user_vocab_size, truncate, period_in_weeks or CENTURY,
                           max_quantity=max_accepted_quantity, min_visits=min_visits, min_baskets=min_baskets,
                           min_average_spend=min_average_spend)

    def augment_raw(self):
        """
        Augment the raw dataset with some extra columns which were figured-out during self.transform_data()
        :return:
        """
        if self.data is None:
            raise NotImplementedError("First transform the raw data")
        if 'product_token' in self.raw_data.columns:
            logging.debug("Raw data is already augmented with extra preprocessed columns")
            return

        raw_data = pd.merge(self.raw_data, self.data[
            ['Invoice', 'StockCode', 'Customer ID', 'product_token', 'user_token', 'invoice_token', 'period_token']],
                            on=['Invoice', 'StockCode', 'Customer ID'], how='inner')
        raw_data[['product_token', 'user_token', 'period_token']] = raw_data[['product_token', 'user_token', 'period_token']].fillna(0)
        raw_data['product_token'] = np.int32(raw_data['product_token'])
        raw_data['period_token'] = np.int32(raw_data['period_token'])
        raw_data['user_token'] = np.int32(raw_data['user_token'])
        self.raw_data = raw_data
        return raw_data


def transform_data(data, stock_vocab_size, user_vocab_size, truncate, period_weeks, max_quantity=1023,
                   min_visits=0, min_baskets=0, min_average_spend=0,
                   remove_singleton_baskets=True, price_segments=1):
    data = _drop_unneeded(data)
    data = _clean_bad_p_q(data)

    data['Period'] = data.Week // period_weeks  # a period is generally ideally 4 weeks long
    n_periods = data.Period.nunique()
    logging.info(f"Breaking the data into {n_periods} period(s) of length {period_weeks} weeks.")

    data = _aggregate_observationally_eq_lines(data)

    if remove_singleton_baskets:
        data = _remove_singletons(data)
    if price_segments > 1:
        data = _segment_prices(data, price_segments=price_segments)

    data = _remove_infrequent(data, min_visits, min_baskets, min_average_spend)

    data['product_token'], stockcode_counts = _index_vocab(data.StockCode, stock_vocab_size, 'product')
    data['user_token'], user_counts = _index_vocab(data['Customer ID'], user_vocab_size, 'user')
    data['period_token'], period_counts = _index_vocab(data.Period, n_periods + 1, 'period')

    invoice_le = LabelEncoder()
    data['invoice_token'] = invoice_le.fit_transform(data['Invoice'])

    if stockcode_counts['UNK'] > 0:
        data = _aggregate_unks(data)
    data = _handle_large_quantities(data, max_quantity, truncate)

    invoices = np.unique(data.Invoice, return_counts=False)

    return data, len(invoices), stockcode_counts, n_periods + 1, len(user_counts), data.Quantity.max()


def _remove_infrequent(data, min_visits=0, min_baskets=0, min_average_spend=0):
    if min_visits == 0 and min_baskets == 0 and min_average_spend == 0:
        return data

    frequent_prods = data.groupby('StockCode').Invoice.nunique() >= min_baskets
    if min_baskets:
        logging.info(f'Removing {len(frequent_prods) - sum(frequent_prods)} products that show up in < {min_baskets} baskets.')
    thinned_data = data.set_index('StockCode').loc[frequent_prods].reset_index().copy()

    thinned_data['Cost'] = thinned_data.Price * thinned_data.Quantity
    checkouts = thinned_data.groupby(['Customer ID', 'Invoice'], as_index=False).Cost.sum()
    thinned_data.drop(columns='Cost')

    customers = checkouts.groupby('Customer ID')
    spenders = customers.Cost.mean() >= min_average_spend

    if min_average_spend:
        logging.info(f'Removing {len(spenders) - sum(spenders)} customers who still spent < ${min_average_spend} '
                     f'per visit.')

    frequents = customers.Invoice.nunique() >= min_visits
    if min_visits:
        logging.info(f'Removing {len(frequents) - sum(frequents)} customers still with < {min_visits} '
                     f'visits in the data.')

    good_custs = spenders & frequents
    thinned_data = thinned_data.set_index('Customer ID').loc[good_custs].reset_index()
    return thinned_data


def _drop_unneeded(data):
    return data.loc[:, ['StockCode', 'Invoice', 'Quantity', 'Price', 'Customer ID', 'Week']]


def _handle_large_quantities(data, max_quantity, truncate):
    __sdq = sum(data.Quantity > max_quantity)
    __q_report = f'Among {len(data)} purchase records, {__sdq} have quantity > {max_quantity}. '
    if truncate:
        logging.info(__q_report + f"Replacing these with {max_quantity}.")
        data['Quantity'] = data['Quantity'].where(data['Quantity'] <= max_quantity, max_quantity)
    else:
        logging.info(__q_report + f"Removing these entries.")
        data = data[data.Quantity <= max_quantity]
    return data


def _index_vocab(series, vocab_size, unit):
    token, counts = index_words(series, vocab_size)
    logging.info(f"Track the {len(set(counts) - {'UNK'})} most frequent {unit}s, "
                 f"the least-viewed of which appears in {list(counts.items())[-1][-1]} records.")
    if counts['UNK'] > 0:
        logging.info(f"{counts['UNK']} purchases regard other {unit}s that are even more rarely seen. "
                     f"Mark these as UNK(nown).\n")
    else:
        logging.info(f"Since vocab_size was larger than the observed number of unique units, we have "
                     f"encoded every unit, and there is no UNK.")
    return np.int32(token), counts


def _clean_bad_p_q(data):
    logging.info(f"Of {len(data)} purchase records, removing {sum(data.Price == np.inf)} with infinite prices "
                 f"and {sum(pd.isnull(data.Price))} missing prices.")
    data = data[data['Price'] < np.inf]
    good_pq = (data['Price'] > 0) & (data['Quantity'] > 0)
    logging.info(
        f"Of {len(data)} purchase records, removing {sum(~good_pq)} with zero or -ve quantity/price.\n")
    return data[good_pq]


def _aggregate_observationally_eq_lines(data):
    """
    :param data:
    We aggregate observationally equivalent lines of data.
    We take the opportunity to calculate mean prices across each period and across the whole data
    :return: out - an adjusted version of data
    """
    assert (data.Quantity > 0).all()
    # We sum the quantities and mean the prices for duplicate invoice/stockcode pairs
    gb_si = data.groupby(['Invoice', 'StockCode', 'Period', 'Customer ID'])
    items_in_period = data.groupby(['StockCode', 'Period'])
    items = data.groupby('StockCode')
    logging.info(f"After aggregation, {len(gb_si)} unique purchase records remain."
                 f" {len(items)} separate stock codes were found in the dataset. "
                 f"On average each is mentioned {len(gb_si) // len(items)} times.")

    mean_prices = pd.DataFrame(items.Price.mean())
    mean_prices.columns = ['MeanPrice']

    timely_prices = pd.DataFrame(items_in_period.Price.mean())
    timely_prices.columns = ['PeriodPrice']

    #TODO: taking the flat average of the price here, where better would be to take the quantity-weighted average.
    q_and_p = gb_si.aggregate(dict(Quantity='sum', Price='mean')).reset_index()
    out = q_and_p.join(mean_prices, on='StockCode')
    out = out.join(timely_prices, on=['StockCode', 'Period'])

    # Assume there is no data-quality issue in the sense that each invoice occurs in only one period

    cust_ids = gb_si['Customer ID'].max().fillna(0)
    if (cust_ids == 0).any():
        logging.info(f"There were {sum(cust_ids == 0)} invoices which were missing customer id - these are kept")
        out['Customer ID'] = out['Customer ID'].fillna(0)

    return out


def _remove_singletons(data):
    data_cost = data.copy()
    data_cost['Cost'] = data.Price * data.Quantity
    gb_i = data_cost.groupby(['Invoice'])
    counts = pd.DataFrame(gb_i.StockCode.count())
    counts.columns = ['n_codes']
    data = data.join(counts, on='Invoice')
    logging.info(f"We observe {len(gb_i)} separate invoices (baskets) costing on average "
                 f"${gb_i.Cost.sum().mean():2.2f} with {int(counts.mean())} items.")
    logging.info(f"The most expensive invoice costs ${gb_i.Cost.sum().max():2.2f}. "
                 f"The biggest contains {int(counts.max())} items.")
    logging.info(
        f'Of {len(data)} purchase records, we remove {sum(data.n_codes <= 1)} which describe singleton baskets.')
    return data[data.n_codes > 1]


def _segment_prices(df, price_segments=4):
    gb = df.groupby('StockCode')
    initial_stock_codes = len(gb)
    quantiles = gb['Price'].apply(lambda x: pd.qcut(x, price_segments, False, duplicates="drop")).astype(str)
    df['StockCode'] = df['StockCode'].astype(str) + quantiles
    final_stock_codes = len(df.groupby('StockCode'))
    logging.info(f'We segmented stock codes in to up to {price_segments} price quantiles per stock code, this resulted ' \
                 f'in the creation of {final_stock_codes - initial_stock_codes} new effective, price segmented stock ' \
                 f'codes')
    return df


def _aggregate_unks(df):
    # Now aggregate any untracked items appearing in a basket, into one UNK:
    unks = (df['product_token'] == 0)
    data_agg = df.copy()
    data_agg['Price'] = df.Price * df.Quantity
    data_agg['Quantity'] = 1
    # data_agg['MeanPrice'] = np.nan
    data_agg[~unks] = df[~unks]
    gb = data_agg.groupby(['Invoice', 'StockCode'])
    data_agg = gb.max()
    data_agg['Price'] = gb.Price.sum()
    logging.info("After aggregating any untracked items in each basket into one UNK, "
                 f"{len(data_agg)} observations remain.")
    return data_agg.reset_index()
