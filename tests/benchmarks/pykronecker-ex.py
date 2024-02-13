import numpy as np
import jax.numpy as jnp
from pykronecker import KroneckerProduct

As = []
for i in range(0,10):
    As.append(jnp.asarray(np.random.normal(size=(4,4))))
x = jnp.asarray(np.random.normal(size=(320, 4**10)))

KP = KroneckerProduct(As)
for i in range(10):
    y = x @ KP
