import numpy as np

from rube.data.clean import RawDataCleaner
from rube.data.generator import Generator


class RealDataGenerator(Generator):
    def __init__(self, batch_size, neg_samples, seed=None, n_lines=None, begin_week=0, end_week=None, max_accepted_quantity=32,
                 stock_vocab_size=2000, user_vocab_size=2000, period_in_weeks=None, truncate=True, save_raw=False,
                 min_visits=0, min_baskets=0, min_average_spend=0, **kwargs):
        """
        :param batch_size:
        :param neg_samples: the number of negative samples to put in Signal Set
        :param seed: (int) random seed for jax
        :param n_lines: (int) take head(n_lines) of data if provided, else take all availible data
        :param begin_week: (int) start after _ weeks of data
        :param end_week: (int) go up to _th week of data
        :param max_accepted_quantity: (int) if truncate is false remove all entries with more than max_accepted_quantity
                                            else truncate the quantity of all entries with > maq to maq
        :param stock_vocab_size: (int) maximum number of products to encode
        :param user_vocab_size: (int) maximum number of users to encode
        :param period_in_weeks: we have the option of breaking the dataset into periods each of length period_in_weeks.
                                Setting a default here of None should ensure that the default will be one period.
        :param truncate: (bool) see max_accepted_quantity
        :param save_raw: (bool) if true then create an attribute called self.raw_data and save the raw data there
        :param min_visits: minimum number of times a customer must show up in the data set to be included in cleaned data
        :param min_baskets: minimum number of baskets a product must show up in in order to be included in the cleaned data
        :param min_average_spend: minimum average spend of a customer to be included in the cleaned data
        """
        raw = self.import_dataset(begin_week=begin_week, end_week=end_week, n_lines=n_lines)

        cleaner = RawDataCleaner(raw)
        cleaner.transform_data(max_accepted_quantity=max_accepted_quantity,
                               stock_vocab_size=stock_vocab_size,
                               user_vocab_size=user_vocab_size,
                               min_visits=min_visits, min_baskets=min_baskets, min_average_spend=min_average_spend,
                               truncate=truncate,
                               period_in_weeks=period_in_weeks)

        t = populate_t(cleaner.data, cleaner.n_samples)
        preprocessed_data = {'q': populate_q(cleaner.data, cleaner.n_samples, len(cleaner.stock_vocab)),
                             'p': populate_p(cleaner.data, cleaner.n_samples, len(cleaner.stock_vocab), t),
                             't': t,
                             'u': populate_u(cleaner.data, cleaner.n_samples)}

        super(RealDataGenerator, self).__init__(preprocessed_data, batch_size, neg_samples, cleaner.max_seen_q,
                                                stock_vocab=cleaner.stock_vocab, n_periods=cleaner.n_periods,
                                                user_vocab_size=cleaner.n_users,
                                                seed=seed, **kwargs)

        self.raw_data = None
        if save_raw:
            self.raw_data = cleaner.augment_raw()

    def import_dataset(self, n_lines):
        raise NotImplementedError


def populate_t(data, n_samples):
    t = np.zeros((n_samples,), dtype=np.int8)
    t[data['invoice_token']] = np.int8(data['period_token'])
    return t


def populate_q(data, n_samples, stock_vocab_size):
    q = np.zeros((n_samples, stock_vocab_size), dtype=np.int8)
    q[data['invoice_token'], data['product_token']] = np.int8(data['Quantity'])
    return q


def populate_p(data, n_samples, stock_vocab_size, t):
    p = np.zeros((n_samples, stock_vocab_size), dtype=np.float32)
    # To start, fill in prices with the mean price for that good
    p[0, data['product_token']] = data['MeanPrice']
    p[1:] = p[0]

    # Fill in timely prices where possible
    for pt in data['period_token'].unique():
        period_data = data[data['period_token'] == pt]
        period_p = p[t == pt]
        period_p[0, period_data['product_token']] = period_data['PeriodPrice']
        period_p[1:] = period_p[0]
        p[t == pt] = period_p

    # Fill in the actual price where it is available
    p[data['invoice_token'], data['product_token']] = data['Price']
    return p


def populate_u(data, n_samples):
    u = np.zeros((n_samples, 1))
    u[data['invoice_token'], :] = data['user_token'][:, np.newaxis]
    u = u.astype(np.int32)
    return u