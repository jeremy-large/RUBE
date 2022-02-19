import jax
import numpy as np
from jax import numpy as jnp

import rube.model
from rube.data.generator import build_signal_set


def build_seed_basket(cleaner, key, basket_size=6, n_products=None):
    df = cleaner.raw_data
    user_token = jax.random.choice(key, df['user_token'].unique().shape[0])[jnp.newaxis]
    vocab_size = n_products or len(cleaner.stock_vocab)
    basket = jnp.zeros(vocab_size)
    basket_tokens = jax.random.choice(key, np.arange(1, n_products), (basket_size,), replace=False)
    basket = basket.at[basket_tokens].set(jax.random.choice(key, jnp.arange(1, cleaner.max_seen_q), (basket_size,)))
    prices = np.zeros((1, len(cleaner.stock_vocab)), dtype=np.float32)

    prices[:, cleaner.data['product_token']] = cleaner.data['MeanPrice']
    prices[:, cleaner.data['product_token']] = cleaner.data['Price']
    if n_products:
        prices = prices[:, :n_products]
    return user_token, basket, prices


def propose_new(user_token, basket, prices, raw_params, key, max_q):
    """
    :param user_token:
    :param basket: we are interested in proceeding from this basket to another, following a Markov chain
    which settles down into an ergodic distribution equal to the true distribution of baskets.
    :param prices: the prices of the goods which are in force, a jnp array.
    :param raw_params: model parameters
    :param key: jaxkey
    :param max_q: biggest permitted quantity that can be drawn
    :return: (the next basket, its utility, indicator variable for whether it differs from its predecessor).
    """
    choices = build_signal_set(basket, key, max_q, 1, replace=False)
    utilities = rube.model.model.qua_model(raw_params, choices, prices, user_token)
    ratio = jnp.exp(utilities[1] - utilities[0])
    if ratio > 1:
        return choices[1], utilities[1], 1

    rand = np.random.uniform()
    idx = 1 if rand < ratio else 0
    return choices[idx], utilities[idx], idx


def generate_draws(params, max_q, draw_key, u, basket, prices, max_iters=10000000, min_iters=2500, sample_freq=50):
    """
    :param params: model parameters
    :param max_q: biggest permitted quantity that can be drawn
    :param draw_key: jaxkey
    :param u: user token. So far, this has been constant and equal to zero in applications.
    :param basket: we are interested in proceeding from this basket to another, following a Markov chain
    which settles down into an ergodic distribution equal to the true distribution of baskets.
    :param prices: the prices of the goods which are in force, a jnp array.
    :param max_iters: the generator will refuse to go on after this many iterations.
    :param min_iters: it is deemed that the ergodic distribution has not been reached until after this many iterations.
    :param sample_freq:
    :yield: The next simulated datapoint
    """
    params = params.copy()
    params['A_'] = params['A_'][:basket.shape[0]]
    for i in range(max_iters):
        draw_key, subkey = jax.random.split(draw_key)
        basket, ut, idx = propose_new(u, basket, prices, params, subkey, max_q)
        if i % sample_freq == 0 and i > min_iters:
            yield basket, ut, idx, prices
