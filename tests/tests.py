from rube.data.fake import FakeDataGenerator
from rube.model.model import RubeJaxModel, load_params

import jax.numpy as jnp
import logging

BATCH_SIZE = 256
NEG_SAMPLES = 3
CONTEXT_SIZE = 3
MAX_QUANTITY = 10
N_SAMPLES = 1000
N_PERIODS = 101
STOCK_VOCAB_SIZE = 20
EMBEDDING_DIM = 8
EPOCHS = 2
USER_VOCAB_SIZE = 10
SEED = 42


def test_generator():
    dg = FakeDataGenerator(batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES, context_size=CONTEXT_SIZE,
                           max_quantity=MAX_QUANTITY, n_samples=N_SAMPLES, stock_vocab=range(STOCK_VOCAB_SIZE),
                           n_periods=N_PERIODS, generate_users=False, seed=SEED)
    x = next(dg)
    assert x[0]['quantity'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, STOCK_VOCAB_SIZE)
    assert x[0]['prices'].shape == (BATCH_SIZE, 1, STOCK_VOCAB_SIZE)
    assert x[1]['output_1'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, 1)


def test_generator_with_user():
    dg = FakeDataGenerator(batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES, context_size=CONTEXT_SIZE,
                           max_quantity=MAX_QUANTITY, n_samples=N_SAMPLES, stock_vocab=range(STOCK_VOCAB_SIZE),
                           n_periods=N_PERIODS, generate_users=True, user_vocab_size=USER_VOCAB_SIZE, seed=SEED)
    x = next(dg)
    assert x[0]['quantity'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, STOCK_VOCAB_SIZE)
    assert x[0]['prices'].shape == (BATCH_SIZE, 1, STOCK_VOCAB_SIZE)
    assert x[0]['users'].shape == (BATCH_SIZE, 1)
    assert x[1]['output_1'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, 1)


def test_model():
    dg = FakeDataGenerator(batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES, context_size=CONTEXT_SIZE,
                           max_quantity=MAX_QUANTITY, n_samples=N_SAMPLES, stock_vocab=range(STOCK_VOCAB_SIZE),
                           n_periods=N_PERIODS, seed=SEED)
    model = RubeJaxModel(stock_vocab_size=STOCK_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, n_periods=N_PERIODS, seed=SEED)
    model.training_loop(dg, epochs=EPOCHS)
    params = load_params(model.params)

    assert jnp.isclose(jnp.array([params['A'].sum(),
                                  params['b'].sum(),
                                  params['c'].sum(),
                                  params['d_1'][0, 0],
                                  params['d_2'][0, 0],
                                  params['d_3'][0, 0]]),
                       jnp.array([1.9843567e+01,  6.7489460e-02, -4.6566129e-09, 1.3051327, 1.3304417, 9.5431876e-01])
                       ).all()


def test_model_with_users():
    dg = FakeDataGenerator(batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES, context_size=CONTEXT_SIZE,
                           max_quantity=MAX_QUANTITY, n_samples=N_SAMPLES, stock_vocab=range(STOCK_VOCAB_SIZE),
                           n_periods=N_PERIODS, generate_users=True, user_vocab_size=USER_VOCAB_SIZE, seed=SEED)
    model = RubeJaxModel(stock_vocab_size=STOCK_VOCAB_SIZE, user_vocab_size=USER_VOCAB_SIZE, n_periods=N_PERIODS,
                         embedding_dim=EMBEDDING_DIM, seed=SEED)
    model.training_loop(dg, epochs=EPOCHS)
    params = load_params(model.params)

    assert jnp.isclose(jnp.array([params['A'].sum(),
                                  params['b'].sum(),
                                  params['c'].sum(),
                                  params['d_1'][0, 0],
                                  params['d_2'][0, 0],
                                  params['d_3'][0, 0]]),
                       jnp.array([2.0080568e+01, 3.9418225, 1.9650906e-07, 1.0503459, 1.0530291, 1.1389079])
                       ).all()
