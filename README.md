# RUBE
* Random Utility-Based Embeddings

This code is intended to implement the algorithms in Lanier, Large and Quah (2022).

___________

Whether online, in a supermarket, or elsewhere,
people now assemble consumption bundles from an extremely wide variety of goods.
This may be modeled as a discrete choice between bundles,
to maximize a random utility depending on the unobserved attributes of the goods in the bundle.

Perhaps attributes may be orders-of-magnitude fewer than goods -
much reducing the effective consumption space.

This code uses stochastic gradient descent across batched samples of a big dataset.
Notwithstanding these features, it is intended to implement at scale a theory of
consistent estimation which appears in  Lanier, Large and Quah (2022).
Thus it estimates every purchased good's latent attributes,
jointly with every consumer's preferences over attributes.

This techniques has similarities to negative-sampling for word (and other) embedding in machine learning.
