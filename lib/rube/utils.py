import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from rube.model.model import load_params, positivize


def normalise_matrix(mat):
    lens = np.sqrt(np.sum(mat * mat, axis=1))[:, None]
    mat_norm = mat / lens
    return mat_norm


@jax.jit
def cosine_similarity(sc1, sc2, mat):
    try:
        return jnp.dot(mat[sc1], mat[sc2])
    except KeyError:
        print(f'Invalid stock code: {sc2}')
        return 1e10


def nearest_neigbours(stock_code, dg, model):
    df = dg.raw_data.copy()
    df = df[df.Quantity > 0]
    scs = df[df.StockCode == stock_code]['product_token'].unique()
    if len(scs) > 1:
        raise ValueError(f'Multiple tokens for stock code {stock_code}: {scs}')
    sc1, = scs
    if sc1 == 0:
        raise ValueError('Selected item was encoded as UNK')
    df.set_index('product_token', inplace=True)
    distance_vmap = jax.vmap(cosine_similarity, in_axes=(None, 0, None))
    params = positivize(load_params(model.params))

    tokens = np.unique(df.index.values)
    distances = distance_vmap(sc1, tokens, normalise_matrix(params['A']))
    distance_df = pd.DataFrame([tokens, distances]).T
    distance_df.columns = ['product_token', 'Similarity']
    distance_df.set_index('product_token', inplace=True)

    df = df.join(distance_df, how='left')
    return df.sort_values(by='Similarity', ascending=False).reset_index().set_index('StockCode')


