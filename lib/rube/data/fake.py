import numpy as np

from rube.data.generator import Generator


class FakeDataGenerator(Generator):
    def __init__(self, batch_size, neg_samples, context_size, max_quantity, n_samples, stock_vocab, n_periods,
                 generate_users=False, user_vocab_size=None, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        data = self.build_samples(n_samples, len(stock_vocab), n_periods, user_vocab_size, context_size, max_quantity,
                                  generate_users)
        super(FakeDataGenerator, self).__init__(data, batch_size, neg_samples, max_quantity,
                                                stock_vocab=stock_vocab, n_periods=n_periods,
                                                user_vocab_size=user_vocab_size, seed=seed, **kwargs)

    def build_samples(self, n_samples, stock_vocab_size, n_periods, user_vocab_size, context_size, max_quantity, generate_users):
        q = np.zeros((n_samples, stock_vocab_size), dtype=np.float32)
        p = np.zeros((n_samples, stock_vocab_size), dtype=np.float32)
        t = np.zeros((n_samples, 1), dtype=np.int32)
        for i in range(n_samples):
            items = np.random.randint(low=1, high=context_size + 1)
            idx = np.random.choice(np.array(list(range(stock_vocab_size,))), size=(items,), replace=False)
            qs = np.random.randint(low=1, high=max_quantity, size=(idx.shape[0]))

            for (j, k) in zip(idx, list(range(idx.shape[0]))):
                q[i, j] = qs[k]

            p[i, :] = np.random.rand(1, stock_vocab_size)
            t[i] = np.random.randint(1, n_periods)

        if generate_users:
            u = np.random.randint(low=1, high=user_vocab_size, size=(n_samples, 1))
            return {'q': q, 'p': p, 't': t, 'u': u}
        else:
            return {'q': q, 'p': p, 't': t}