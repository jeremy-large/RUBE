import numpy as np

from rube.data.clean import RawDataCleaner
from rube.data.generator import Generator


class RealDataGenerator(Generator):
    def __init__(self, batch_size, neg_samples, seed=None, n_lines=None, max_accepted_quantity=32,
                 stock_vocab_size=2000, user_vocab_size=2000, truncate=True, save_raw=False, **kwargs):
        """
        :param batch_size:
        :param neg_samples: the number of negative samples to put in Signal Set
        :param seed: (int) random seed for jax
        :param n_lines: (int) take head(n_lines) of data if provided, else take all availible data
        :param max_accepted_quantity: (int) if truncate is false remove all entries with more than max_accepted_quantity
                                            else truncate the quantity of all entries with > maq to maq
        :param stock_vocab_size: (int) maximum number of products to encode
        :param user_vocab_size: (int) maximum number of users to encode
        :param truncate: (bool) see max_accepted_quantity
        :param save_raw: (bool) if true then create an attribute called self.raw_data and save the raw data there
        """
        raw = self.import_dataset(n_lines)

        cleaner = RawDataCleaner(raw)
        cleaner.transform_data(max_accepted_quantity=max_accepted_quantity,
                               stock_vocab_size=stock_vocab_size,
                               user_vocab_size=user_vocab_size,
                               truncate=truncate)

        preprocessed_data = {'q': populate_q(cleaner.data, cleaner.n_samples, len(cleaner.stock_vocab)),
                             'p': populate_p(cleaner.data, cleaner.n_samples, len(cleaner.stock_vocab)),
                             'u': populate_u(cleaner.data, cleaner.n_samples)}

        super(RealDataGenerator, self).__init__(preprocessed_data, batch_size, neg_samples, cleaner.max_seen_q,
                                                stock_vocab=cleaner.stock_vocab, user_vocab_size=cleaner.n_users,
                                                seed=seed, **kwargs)

        self.raw_data = None
        if save_raw:
            self.raw_data = cleaner.augment_raw()

    def import_dataset(self, n_lines):
        raise NotImplementedError


def populate_q(data, n_samples, stock_vocab_size):
    q = np.zeros((n_samples, stock_vocab_size), dtype=np.int8)
    q[data['invoice_token'], data['product_token']] = np.int8(data['Quantity'])
    return q


def populate_p(data, n_samples, stock_vocab_size):
    p = np.zeros((n_samples, stock_vocab_size), dtype=np.float32)
    # To start, fill in prices with the mean price for that good
    p[0, data['product_token']] = data['MeanPrice']
    p[1:] = p[0]
    # Fill in the actual price where it is available
    p[data['invoice_token'], data['product_token']] = data['Price']
    return p


def populate_u(data, n_samples):
    u = np.zeros((n_samples, 1))
    u[data['invoice_token'], :] = data['user_token'][:, np.newaxis]
    u = u.astype(np.int32)
    return u