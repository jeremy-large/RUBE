import csv
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm_text
from tqdm.notebook import tqdm as tqdm_notebook
import logging
import pickle
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam, unpack_optimizer_state, pack_optimizer_state

DEFAULT_KEY = jax.random.PRNGKey(42)


class RubeJaxModel:
    def __init__(self, stock_vocab_size, embedding_dim, n_periods=1, step_size=0.01, user_vocab_size=1, fn='qua', load_model=None, seed=None):
        '''
        :param stock_vocab_size: (int) maximum number of products to encode, must match argument by the same name passed
                                       to the data generator
        :param embedding_dim: (int) size of the embedding dimension
        :param n_periods: (int) the number of periods into which to break the model - helps handle demand endogeneity
        :param step_size: (float) step size for the adam optimiser
        :param user_vocab_size: (int) Maximum number of users to encode, 1 implies no user specific parameters, otherwise
                                      must match the argument by the same name passed to the data generator
        :param fn: (string or function) utility function to apply
        :param load_model: (string or None) if not None, load file stored at string containing an optimiser state (for hot start)
        :param seed: (int) random seed for jax
        '''
        self.stock_vocab_size = stock_vocab_size
        self.embedding_dim = embedding_dim
        self.n_periods = n_periods
        self.user_vocab_size = user_vocab_size
        self.model = fn if callable(fn) else qua_model if fn == 'qua' else old_model
        self.opt_init, self.opt_update, self.get_params = adam(step_size=step_size)
        if load_model is not None:
            saved_state = pickle.load(open(load_model, 'rb'))
            self.opt_state = pack_optimizer_state(saved_state)
            self.params = self.get_params(self.opt_state)
        else:
            self.params = self._initialize_model(seed=seed)
            self.opt_state = self.opt_init(self.params)
        self.train_accuracies = []
        self.test_accuracies = []
        self.losses = []

    def _initialize_model(self, seed=None):
        def regen(key):
            _, subkey = jax.random.split(key)
            return subkey

        if seed is None:
            key = DEFAULT_KEY
        else:
            key = jax.random.PRNGKey(seed)

        dim = self.embedding_dim
        R = jnp.eye(dim)
        c = jnp.zeros((dim, self.n_periods))
        b_intercept = jnp.zeros((dim, 1))

        A = jax.random.normal(key, (self.stock_vocab_size, dim)) / dim
        key = regen(key)

        b = jax.random.normal(key, (dim, self.user_vocab_size)) / dim
        key = regen(key)

        d_1 = jax.random.exponential(key, (1, self.user_vocab_size)) / dim
        key = regen(key)

        d_2 = jax.random.exponential(key, (1, self.user_vocab_size)) / dim
        key = regen(key)

        d_3 = jax.random.exponential(key, (1, self.user_vocab_size)) / dim

        return {'A_': A, 'R': R, 'lb_': b, 'b_c_':b_intercept, 'c_': c, 'ld_1': d_1, 'ld_2': d_2, 'ld_3': d_3}

    def model_predict(self, x):
        qs = x[0]['quantity']
        p  = x[0]['prices']
        t  = x[0]['period']
        u  = x[0]['users'] if 'users' in x[0].keys() else jnp.zeros((qs.shape[0], 1), dtype=jnp.int8)
        logits = jax.vmap(self.model, in_axes=(None, 0, 0, 0, 0))(self.params, qs, p, t, u)
        return jax.nn.softmax(logits, axis=1)

    def accuracy(self, x):
        """
        :param x: data structure containing one batch, usable for fitting with self.update()
        :return: accuracy of the fitted model against x
        """
        _biggest = lambda a: jnp.argmax(a, axis=1)
        labels = x[1]['output_1']
        return jnp.mean(_biggest(self.model_predict(x)) == _biggest(labels))

    def update(self, step, x):
        loss, grads = jax.value_and_grad(model_loss)(self.get_params(self.opt_state), x, self.model)
        # Make sure grads are not nan because these are propagated
        grads = {key: jnp.nan_to_num(grads[key]) for key in grads.keys()}
        self.opt_state = self.opt_update(step, grads, self.opt_state)
        self.params = self.get_params(self.opt_state)
        self.train_accuracies.append(self.accuracy(x))
        self.losses.append(loss)

    def training_loop(self, gen, epochs=5, fit_dir=None, notebook=False, verbosity=5):
        '''
        :param gen: an instance of rube.data.generator.Generator
        :param epochs: number of epochs (iterations over entire dataset) to use
        :param fit_dir: (optional) location to save (partially) fit models
        :param notebook: if True, then use progress bar for notebooks, else text progress bar
        :param verbosity: log metrics every `verbosity` epochs.
        '''
        logging.info(f"Holdout accuracy at first (would be 1/(ns+1) in costless data w/ unit q's): {self.accuracy(gen.holdout):2.3f}")
        tqdm = tqdm_notebook if notebook else tqdm_text
        step = 0
        n_iter = gen.get_n_iter()
        logging.info(f'Metrics that will be displayed every {verbosity} epochs: '
                     f'TA = train accuracy, HA = holdout accuracy, TL = train loss')
        for epoch in range(epochs):
            for batch in tqdm(gen, total=n_iter):
                step += 1
                self.update(step, batch)
            self.test_accuracies.append(self.accuracy(gen.holdout))
            if (epoch + 1 == epochs) or (epoch % verbosity) == 0:
                logging.info(f'Epoch {epoch + 1}/{epochs}: '
                             f'TA= {sum(self.train_accuracies[-n_iter:]) / n_iter:2.3f}, '
                             f'HA= {self.test_accuracies[-1]:2.3f}, '
                             f'TL= {self.losses[-1]:2.3f}')
            if fit_dir:
                trained_params = unpack_optimizer_state(self.opt_state)
                loss_df = pd.DataFrame(self.losses, columns=['loss']).groupby(np.arange(n_iter * (epoch + 1)) // n_iter).mean()
                train_acc_df =  pd.DataFrame(self.train_accuracies, columns=['train_accuracy']).groupby(np.arange(n_iter * (epoch + 1)) // n_iter).mean()
                test_acc_df = pd.DataFrame(self.test_accuracies, columns=['test_accuracy'])
                epoch_df = pd.DataFrame(np.arange(epoch + 1), columns=['epoch'])
                metrics_df = pd.concat((epoch_df, loss_df, train_acc_df, test_acc_df), axis=1)
                metrics_df.to_csv(fit_dir / 'metrics.csv', index=False)

                f = fit_dir / f"epoch_{epoch}.pkl"
                logging.debug(f'Saving (partially) trained model to {f}')
                pickle.dump(trained_params, open(f, "wb"))

                f = fit_dir / f"final_model.pkl"
                logging.debug(f'Saving provisional final trained model to {f}')
                pickle.dump(trained_params, open(f, "wb"))
                save_embeddings_tsv(positivize(load_params(self.params)),
                                    gen, 'StockCode',
                                    embedding_file=fit_dir / 'A_embeddings.tsv',
                                    vocab_file=fit_dir / 'A_vocab.tsv')

@jax.jit
def qua_model(raw_pars, q, p, t, u):
    '''
    :param raw_pars: Raw params of the qua model (A_, R, lb, ld_*)
    :param q: (negative_samples, stock_vocab_size) array of quantities
    :param p: (1, stock_vocab_size) array of prices
    :param u: (1,) vector containing a user_id, if no user data then it should be 0
    :param t: (1,) vector containing a market_id, if no market segmentation, data then it should be 0
    :return: (1, negative_samples) vector of utilities
    '''
    params = load_params(raw_pars)
    A = params['A']
    b = params['b'][:, u]
    c = params['c'][:, t]
    d_1 = params['d_1'][:, u]
    d_2 = params['d_2'][:, u]
    d_3 = params['d_3'][:, u]

    a = jnp.dot(q, A)

    aa = jnp.diag(jnp.dot(a, a.T))

    ab_plus_ac = jnp.dot(a, b + c).T

    m = jnp.dot(p, q.T)[0]

    util = ab_plus_ac - aa - (d_1 * m) - (d_2 * m * m) - (2 * d_3 * a[:, 0] * m)
    return util.T


def positivize(params):
    flipper = jnp.diag(jnp.sign(params['b'].mean(axis=1)))
    # we can flip everything except the first embedding dimension:
    flipper = flipper.at[0, 0].set(1)
    A = jnp.dot(params['A'], flipper)
    b = jnp.dot(flipper, params['b'])
    params.update({'A': A, 'b': b})

    if 'c' in params:
        c = jnp.dot(flipper, params['c'])
        params.update({'c': c})

    return params


@jax.jit
def load_params(raw_pars):
    A = jnp.dot(raw_pars['A_'], raw_pars['R'])
    # first embedding dimension forced positive:
    A = A.at[:, 0].set(jnp.exp(A[:, 0]))
    # don't embed UNK:
    A = A.at[0].set(0)
    b = raw_pars['lb_']
    K1 = (b.shape[0], 1)
    b = b + raw_pars.get('b_c_', jnp.zeros(K1))

    # per_period coefficients (c) were not specified in earlier versions, so default these to zero:
    c = raw_pars.get('c_', jnp.zeros(K1))
    # since the first column of c refers to the UNK period (normally unpopulated), we exclude this now:
    c_mean = jnp.mean(c[:, 1:], axis=1)[:, None]
    c = c - c_mean # normalize so c sums to zero and UNK is zero
    c = c.at[:, 0].set(0)

    d_1 = jnp.exp(raw_pars['ld_1'])
    d_2 = jnp.exp(raw_pars['ld_2'])
    d_3 = jnp.exp(raw_pars['ld_3'])
    return dict(A=A, b=b, c=c, d_1=d_1, d_2=d_2, d_3=d_3)


def psi(params, np_range=None):
    """
    :param params: model parameters
    :param np_range: an optional np.arange() object to home-in on certain products
    :return: generate the identified substrate of the parameters, denoted psi in the paper
    """
    A = params['A']
    if np_range is not None:
        A = A[np_range]
    b = params['b']
    return (A @ A.T, A @ b, params['d_1'], params['d_2'], params['d_3'].T * A[:, 0])


@jax.jit
def loss(logits, labels):
    norm_logits = jax.nn.log_softmax(logits, axis=1)
    loss = jnp.mean(-norm_logits[jnp.nonzero(labels, size=logits.shape[0])])
    return loss


@jax.tree_util.Partial(jax.jit, static_argnums=2)
def model_loss(params, x, model):
    # partial tells jax that we want to "trace" params and x because they will change
    # but we tell it that model (arg #2) is static meaning jax can assume it won't
    # change over calls of this function.
    qs = x[0]['quantity']
    p  = x[0]['prices']
    t  = x[0]['period']
    u  = x[0]['users'] if 'users' in x[0].keys() else jnp.zeros((qs.shape[0], 1), dtype=jnp.int8)
    labels = x[1]['output_1']
    # in_axes tells jax which dimension is the batch dimension for each argument
    # None implies that there is no batch dimension for that argument (so use the
    # same value for each iteration)
    logits = jax.vmap(model, in_axes=(None, 0, 0, 0, 0))(params, qs, p, t, u)
    batch_loss = loss(logits, labels)

    return batch_loss


def save_embeddings_tsv(params, dg, embedding_file='A_embeddings.tsv', vocab_file='A_vocab.txt'):
    with open(embedding_file, 'w+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerows(params['A'])

    with open(vocab_file, 'w') as tsvfile:
        writer = csv.writer(tsvfile)
        for row in list(dg.stock_vocab):
            writer.writerow([row,])
