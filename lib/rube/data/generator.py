import jax
import jax.numpy as jnp
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import logging


class Generator:
    def __init__(self, data, batch_size, neg_samples, max_quantity, stock_vocab, n_periods, replace=True, user_vocab_size=None,
                 repeat_holdout=1, seed=None, shuffle=False, test_size=0.02):
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.max_quantity = max_quantity
        self.repeat_holdout = repeat_holdout
        self.replace = replace
        self.test_size = test_size if test_size < 1 else int(test_size)
        effective_test_size = self.calc_effective_test_size(data['q'].shape[0])
        if effective_test_size > self.batch_size:
            logging.warning(f'Effective test size ({effective_test_size}) is bigger than the batch '
                            f'size ({self.batch_size}), as a result it is possible that memory issues will arise, '
                            f'even if the training would otherwise be possible with these parameters.')
        logging.info(f"A proportion {test_size} of the data will be reserved as holdout.")
        if replace:
            logging.warning("The Signal Sets' -ve items are generated with replacement. "
                            "While this is a speedup, it may fail our assumptions for consistency.")
        self.stock_vocab = stock_vocab
        if user_vocab_size is not None:
            self.user_vocab_size = user_vocab_size
            self.contains_user_data = True
        else:
            self.contains_user_data = False
        self.shuffle = shuffle
        self.training_data, self.holdout = self.define_holdout(data)
        self.n_samples = self.training_data['q'].shape[0]
        self.n_periods = n_periods
        self._index = 0
        if seed is not None:
            np.random.seed(seed)

    def calc_effective_test_size(self, nrows):
        if self.test_size >= 1:
            return self.test_size * self.repeat_holdout
        else:
            return nrows * self.test_size

    def get_stock_vocab_size(self):
        return len(self.stock_vocab)

    def get_n_periods(self):
        return self.n_periods

    def get_user_vocab_size(self):
        if self.contains_user_data:
            return self.user_vocab_size
        else:
            raise ValueError("Generator doesn\'t contain user data")

    def get_n_iter(self):
        return np.int32(np.ceil(self.n_samples / self.batch_size))

    def define_holdout(self, data):
        if self.contains_user_data:
            q_train, q_test, p_train, p_test, t_train, t_test, u_train, u_test =\
                train_test_split(data['q'], data['p'], data['t'], data['u'], test_size=self.test_size)
        else:
            q_train, q_test, p_train, p_test, t_train, t_test = \
                train_test_split(data['q'], data['p'], data['t'], test_size=self.test_size)
        if self.repeat_holdout > 1:
            logging.info(f'Since repeat holdout was set, generating {self.repeat_holdout} signal sets for every '
                         f'held-out basket')
        q_test = self.batch_signal_set(np.repeat(q_test, self.repeat_holdout, axis=0), randomize=True)
        p_test = np.repeat(p_test, self.repeat_holdout, axis=0).reshape(q_test.shape[0], 1, -1)
        t_test = np.repeat(t_test, self.repeat_holdout, axis=0).reshape(q_test.shape[0], 1)
        test_target = np.repeat(np.array([1.] + [0. for _ in range(self.neg_samples)]).reshape(1, -1, 1),
                                q_test.shape[0], axis=0)

        train = {'q': q_train, 'p': p_train, 't': t_train}
        test = {'quantity': q_test, 'prices': p_test, 'period': t_test}
        if self.contains_user_data:
            train.update({'u': u_train})
            test.update({'users': np.repeat(u_test, self.repeat_holdout, axis=0)})
        test = test, {'output_1': test_target}

        return train, test

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self.n_samples:
            self._index = 0
            raise StopIteration  # end of one epoch

        # Note that it is possible that less than batch_size observations are left in q. We allow
        # for this by using q.shape[0] instead of self.batch size after q is defined.
        q = self.batch_signal_set(self.training_data['q'][self._index: self._index + self.batch_size])
        p = self.training_data['p'][self._index: self._index + self.batch_size].reshape(q.shape[0], 1, -1)
        t = self.training_data['t'][self._index: self._index + self.batch_size].reshape(q.shape[0], 1)
        target = np.repeat(np.array([1.] + [0. for _ in range(self.neg_samples)]).reshape(1, -1, 1),
                           q.shape[0], axis=0)

        if self.shuffle:
            # Shuffle so the label is not always the first sample in the batch
            for i in range(q.shape[0]):
                q[i, :], target[i, :] = shuffle(q[i, :], target[i, :]) # should we not also shuffle prices here?

        x, y = {'quantity': q, 'prices': p, 'period': t}, {'output_1': target}
        if self.contains_user_data:
            u = self.training_data['u'][self._index: self._index + self.batch_size]
            x['users'] = u

        self._index = self._index + self.batch_size
        return x, y


    def batch_signal_set(self, baskets, randomize=False):
        """
        :param baskets: a collection of baskets: our data
        :param randomize: if true, generate a new random key for every basket, if false than only a new key for every batch
        :return: a batch of signal sets in the form of a 3d array
        """
        if randomize:
            keys = jnp.array([jax.random.PRNGKey(x) for x in np.random.randint(low=1, high=10000000, size=(baskets.shape[0],))])
            return jax.vmap(build_signal_set, in_axes=(0, 0, None, None, None))(jnp.array(baskets),
                                                                                keys,
                                                                                self.max_quantity,
                                                                                self.neg_samples,
                                                                                self.replace)
        else:
            key = jax.random.PRNGKey(np.random.randint(low=1, high=10000000))
            return jax.vmap(build_signal_set, in_axes=(0, None, None, None, None))(jnp.array(baskets),
                                                                                   key,
                                                                                   self.max_quantity,
                                                                                   self.neg_samples,
                                                                                   self.replace)


@jax.jit
def normalise(x):
    return x / x.sum()


@jax.tree_util.Partial(jax.jit, static_argnums=(2, 3, 4))
def build_signal_set(basket, key, max_quantity, neg_samples, replace):
    nonzero = (basket > 0)[1:].astype(jnp.int8)
    poss_neg_samples = (1 - nonzero)
    # we start `rng` at 1 so that we don't pick UNK as a positive or negative item:
    rng = jnp.arange(1, basket.shape[0])
    item_index = jax.random.choice(key, rng, p=normalise(nonzero))
    fake_item_idx = jax.random.choice(key, rng, (neg_samples,), replace=replace, p=normalise(poss_neg_samples))
    fake_item_quantity = jax.random.choice(key, jnp.arange(1, max_quantity+1), replace=True, shape=(neg_samples,))

    bout = jnp.repeat(basket[None, :], neg_samples + 1, axis=0)

    # clear out the positive item from the parts of bout which are to describe negative samples:
    bout = bout.at[1:, item_index].set(0)

    # assign the new negative items to fake baskets (in a vectorised fashion)
    arange = jnp.arange(1, neg_samples + 1)
    bout = bout.at[arange, fake_item_idx].set(fake_item_quantity)

    return bout
