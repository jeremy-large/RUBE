import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rube.data.tokenize import index_words


class RawDataCleaner:
    def __init__(self, raw):
        self.raw_data = raw
        self.data = None
        self.n_samples = None
        self.stock_vocab = None
        self.n_users = None
        self.max_seen_q = None

    def transform_data(self,
                      max_accepted_quantity=32,
                      stock_vocab_size=2000,
                      user_vocab_size=2000,
                      truncate=True):
        """
        :param raw: (dataframe) raw data set
        :param max_accepted_quantity: (int) if truncate is false remove all entries with more than max_accepted_quantity
                                            else truncate the quantity of all entries with > maq to maq
        :param stock_vocab_size: (int) maximum number of products to encode
        :param user_vocab_size: (int) maximum number of users to encode
        :param truncate: (bool) see max_accepted_quantity
        """
        self.data, self.n_samples, self.stock_vocab, self.n_users, self.max_seen_q = \
            transform_data(self.raw_data, stock_vocab_size, user_vocab_size, truncate, max_quantity=max_accepted_quantity)

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
            ['Invoice', 'StockCode', 'Customer ID', 'product_token', 'user_token', 'invoice_token']],
                            on=['Invoice', 'StockCode', 'Customer ID'], how='inner')
        raw_data[['product_token', 'user_token']] = raw_data[['product_token', 'user_token']].fillna(0)
        raw_data['product_token'] = np.int32(raw_data['product_token'])
        raw_data['user_token'] = np.int32(raw_data['user_token'])
        self.raw_data = raw_data
        return raw_data


def transform_data(data, stock_vocab_size, user_vocab_size, truncate, max_quantity=1023,
                   remove_singleton_baskets=True, price_segments=1):
    data = _clean_bad_p_q(data)
    data = _aggregate_observationally_eq_lines(data)
    if remove_singleton_baskets:
        data = _remove_singletons(data)
    if price_segments > 1:
        data = _segment_prices(data, price_segments=price_segments)

    data['product_token'], stockcode_counts = _index_vocab(data.StockCode, stock_vocab_size, 'product')
    data['user_token'], user_counts = _index_vocab(data['Customer ID'], user_vocab_size, 'user')

    invoice_le = LabelEncoder()
    data['invoice_token'] = invoice_le.fit_transform(data['Invoice'])

    if stockcode_counts['UNK'] > 0:
        data = _aggregate_unks(data)
    data = _handle_large_quantities(data, max_quantity, truncate)

    invoices = np.unique(data.Invoice, return_counts=False)

    return data, len(invoices), stockcode_counts, len(user_counts), data.Quantity.max()



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
    logging.info(f"Track the {vocab_size} most frequent {unit}s, "
                 f"the least-viewed of which appears in {list(counts.items())[-1][-1]} baskets.")
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
    data = data[data.Quantity > 0]
    # We sum the quantities and mean the prices for duplicate invoice, stockcode pairs
    gb_si = data.groupby(['Invoice', 'StockCode'])
    gb_s = data.groupby(['StockCode'])
    quantities = pd.DataFrame(gb_si['Quantity'].sum())
    prices = gb_si['Price'].mean()
    mean_prices = pd.DataFrame(gb_s['Price'].mean())
    mean_prices.columns = ['MeanPrice']
    cust_ids = gb_si[['Customer ID']].max().fillna(0)
    data = quantities.join(prices).join(mean_prices, on='StockCode').join(cust_ids).reset_index()
    logging.info(f"{len(mean_prices)} separate stock codes were found in the dataset. "
                 f"On average each is mentioned {len(prices) // len(mean_prices)} times.")
    return data


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
    quantiles = gb['Price'].apply(lambda x: pd.qcut(x, n_quantiles, False, duplicates="drop")).astype(str)
    df['StockCode'] = df['StockCode'].astype(str) + quantiles
    final_stock_codes = len(df.groupby('StockCode'))
    logging.info(f'We segmented stock codes in to up to {n_quantiles} price quantiles per stock code, this resulted ' \
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
