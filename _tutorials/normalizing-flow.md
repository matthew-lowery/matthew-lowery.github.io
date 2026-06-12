---
title: "Normalizing Flow"
order: 1
notebook: /files/tutorials/normalizing_flow.ipynb
excerpt: "RealNVP coupling layers, change of variables, maximum likelihood training, and sampling in JAX and Equinox."
---

In normalizing flows, we parameterize a C^1-diffeomorphic function to transform a random variable $Z$ with a 
given base density to match our data distribution $X$. 

The choice of a such a function enables us to directly estimate the density of the new variable under this transformation $T_\theta: Z \rightarrow X$ with the caveat that the dimensionality of $X$ must match that of $Z$.

$$ p_X(x) = p_Z(T_\theta^{-1}(x)) \left| \det \nabla_x T_\theta^{-1}(x) \right| $$

Consequently, we can directly perform maximum likelihood estimation to learn $T_\theta$ given samples $x \sim X$ (our data).

$$ \begin{align}
\mathcal{L}_\theta &= \mathcal{D}_{KL} [q_\theta || p(x)] \\
&= - \mathbb{E}_{x \sim p}[p_Z(T_\theta^{-1}(x)) \left| \det \nabla_x T_\theta^{-1}(x) \right|] 
\end{align} $$

We can also sample the learned distribution $X$ under the transformation easily by sampling $z \sim Z$ with $x = T_\theta (z)$. 

Also note most authors will directly parameterize the mapping $T_\theta: X \rightarrow Z$ (instead of $T_\theta: Z \rightarrow X$ as is written above) as in [RealNVP](https://arxiv.org/abs/1605.08803) below. The convenience here is that in training we only have to evaluate $T$ and its jacobian, and in sampling we evaluate $T^{-1}$.

Thus, bearing the required computations for training and sampling in mind, layers tend to be designed 
to have cheap to compute inverses and jacobian determinants, 
as the naive determinant computation is cubic in the dimension of $X$. In particular the fact that the determinant of a triangular matrix is the product of its diagonal is often exploited, reducing this complexity to linear. 


### Coupling layers 

exploit this in so much as they design a transformation by spliting a random variable 
into a frozen and active subspace, and having one subspace as input to a function which transforms the other. 

[RealNVP](https://arxiv.org/abs/1605.08803) chooses specifically,

$$\begin{align}
z_1 &= x_1 \\
z_2 &= x_2 \circ \exp(s_\theta (x_1)) + t_\phi(x_1)
\end{align} $$

with $x \in \mathbb{R^D}, x = x_1 \oplus x_2$. (It is usally the case that $x$ is split in half, i.e. $ x_1, x_2 \in \mathbb{R}^{D//2}$).

Then the inverse is trivially,

$$\begin{align}
x_1 &= z_1 \\
x_2 &= (z_2 - t_\phi(z_1)) \circ \exp( - s_\theta (z_1))
\end{align} $$

And the resultant Jacobian is triangular!

$$ \frac{\partial z}{\partial x^T} =
    \begin{bmatrix}
    \frac{\partial z_1}{\partial x_1^T} & \frac{\partial z_1}{\partial x_2^T} \\
    \frac{\partial z_2}{\partial x_1^T} & \frac{\partial z_2}{\partial x_2^T}
    \end{bmatrix}
    = \begin{bmatrix}
    \mathbb{I} & 0 \\
    \frac{\partial z_2}{\partial x_1^T} & \text{diag}(\exp(s(x_1)))
    \end{bmatrix}
    $$


```python
import orthojax as ojax ### easy orthogonal polynomials
from typing import List, Callable, Tuple
import equinox as eqx
import matplotlib.pyplot as plt
import jax
from jax import random as jr, numpy as jnp
from jaxtyping import Float, Array
from jax.scipy.stats import norm, multivariate_normal
import optax
from functools import partial
from abc import ABC, abstractmethod
  
key = jr.PRNGKey(42)
```

As far as the software-engineering aspect of our layers, we use a mask to grab the partition of $x$ we need. We flip back and fourth between partitions we're acting on at each layer. 

Jax's tracing will yell at us if we use a boolean mask when we index in the forward pass, so we convert them to integer indices with the jnp.nonzero() call.



```python
class RealNVPLayer(eqx.Module):
    s: eqx.Module
    t: eqx.Module
    mask: Tuple[jax.Array]
    n_mask: Tuple[jax.Array]
    def __init__(self, *, s_function, t_function, mask, key):

        keys = jr.split(key,) 
        assert mask.dtype == jnp.bool
        in_size, out_size = int((~mask).sum()), int(mask.sum())
        self.s = s_function(in_size=in_size, out_size=out_size, key=keys[0])
        self.t = t_function(in_size=in_size, out_size=out_size, key=keys[1])
        self.mask = jnp.nonzero(mask)
        self.n_mask = jnp.nonzero(~mask)
        
    # The forward function __call__ is the way to sample the model, T: Z -> X. 
    def forward(self, z,):
        x = z.at[self.mask].set((z[self.mask] - self.t(z[self.n_mask])) * jnp.exp(-self.s(z[self.n_mask])))
        return x

    # The inverse function returns both the inverse, T^-1: X -> Z and the log jac det
    def inverse(self, x,): 
        log_jac_det = jnp.sum(self.s(x[self.n_mask]))
        z = x.at[self.mask].set(x[self.mask]*jnp.exp(self.s(x[self.n_mask])) + self.t(x[self.n_mask]))
        return z, log_jac_det 
```

!!! note
    We would generally like our layer to be agnostic to the choice of parameterized functions within them, 
    so its only job is to plug in the in_size (domain dimension), and out_size (the codomain dimension), and 
    other hyperparameters specific to the particular flavor of parameterized function should be plugged in beforehand, i.e. polynomial degree, MLP depth, activation function. 




```python
class PolynomialExpansion(eqx.Module):
    coefs: jax.Array
    _basis: List[Callable]

    def __init__(self, *, in_size, out_size, degree, total_degree, poly_type='hermite', key):
        if poly_type == 'hermite':
            self._basis = ojax.TensorProduct(total_degree,
                                            [ojax.make_hermite_polynomial(deg) for deg in (degree,)*in_size])
        elif poly_type == 'legendre':
            raise NotImplementedError("")
            
        key,_ = jr.split(key)
        self.coefs = jr.normal(key, (self._basis.num_basis, out_size)) * 0.1

    ### basis has jax arrays within it and equinox will train anything that is a jax array so this 
    ### our counter
    @property
    def basis(self):
        return jax.lax.stop_gradient(self._basis)

    def __call__(self, x):
        ### shim [None]/.squeeze() because basis takes a batch dimension that of course we vmap over in jax
        return (self.basis(x[None]) @ self.coefs).squeeze()

```

Now we create model which is just a composition of base layers. 
Fortunately, composing these bijective layers results in another bijective function, with
its inverse as:

$$ T_\theta^{-1} = T^{-1}_1 \circ \dots \circ T^{-1}_{L-1} \circ T^{-1}_L $$

And its log jacobian determinant simply a sum of the individual log jacobian determinants across layers. 


```python
class Normie(eqx.Module):
    base_layers: List[eqx.Module]
    base_dist: eqx.Module
    
    def __init__(self, *, base_layers, base_dist, key):
        self.base_layers = []
        [self.base_layers.append(base_layers[i](key=k)) for i,k in enumerate(jr.split(key, len(base_layers)))]
        self.base_dist = base_dist
        
    def sample(self, key):
        z = self.base_dist.rvs(key, shape=())
        x = z
        for layer in self.base_layers[::-1]:
            x = layer.forward(x)
        return x    

    def train(self, x):
        z = x
        log_jac_det=0.
        for layer in self.base_layers:
            z,log_jac_det_l = layer.inverse(z)
            log_jac_det+=log_jac_det_l
        logp_x = self.base_dist.logpdf(z) + log_jac_det
        return z, logp_x

```

### Experiment Configuration

We aim to learn a randomly initialized distribution $X \sim \mathcal{N}(\mu, LL^T)$, 
 $\mu, L \sim \mathcal{N}(0,I)$, given samples $\{x_i\}_1^N$. 


```python
x_dim = 4
d = x_dim // 2

n_samples = 50000

### prior, x
keys = jr.split(key,3)
L_x = jr.normal(keys[0], (int(x_dim*(x_dim+1)/2)))
L_x = jnp.zeros((x_dim,x_dim)).at[jnp.tril_indices(x_dim)].set(L_x)
x_true_mu = jr.normal(keys[1], (x_dim,))
x_true_cov = L_x @ L_x.T
x_true_samples = jr.multivariate_normal(keys[2], x_true_mu, x_true_cov, shape=(n_samples,)) ### (n_samples, n)
```

## The base distribution $Z$
is commonly set to be 
a standard multivariate normal, and it can also be trained. Logistic distributions are used quite frequently as base distributions also.


```python
class BaseDistribution(ABC):
    @abstractmethod
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def logpdf(self, x):
        pass
    @abstractmethod
    def rvs(self, key, shape):
        pass
        
class FixedStandardNormal(eqx.Module, BaseDistribution):
    dim: int
    def __init__(self, *, dim,):
        self.dim = dim
    def logpdf(self, x):
        return multivariate_normal.logpdf(x, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim))
    def rvs(self, key, shape):
        return jr.multivariate_normal(key, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim), shape=shape)

class TrainableStandardNormal(eqx.Module, BaseDistribution):
    mu: jax.Array
    Sig: jax.Array
    def __init__(self, *, dim, key):
        self.mu = jnp.zeros((dim,))
        self.Sig = jr.uniform(key, (dim,))
        jnp.eye(x_dim)
        
    def logpdf(self, x):
        return multivariate_normal.logpdf(x, mean=self.mu, cov=jnp.diag(jax.nn.softplus(self.Sig)))
    def rvs(self, key, shape):
        return jr.multivariate_normal(key, mean=self.mu, cov=jnp.diag(jax.nn.softplus(self.Sig)), shape=shape)
        
# base_dist = FixedStandardNormal(dim=x_dim)
key,_ = jr.split(key)
base_dist = TrainableStandardNormal(dim=x_dim, key=key)

```

### Choose either MLP or orthogonal Polynomials for base layer's $s_\theta$ and $t_\phi$


```python
### RealNVP
s_function = partial(eqx.nn.MLP, depth=1, width_size=32, activation=jax.nn.tanh)
t_function = s_function

# s_function = partial(PolynomialExpansion, degree=3, total_degree=3)
# t_function = s_function

base_layer = partial(RealNVPLayer, s_function=s_function, t_function=t_function)
```

### Instantiate the Compositional Model


```python
depth = 2
mask = jnp.zeros((x_dim,)).at[:d].set(1.).astype(bool)
base_layers = [partial(base_layer, mask=mask^bool(i%2)) for i in range(depth)] ### xor does mask flipping
model = Normie(base_layers=base_layers, base_dist=base_dist, key=key)
```

### Training/Optimizer Configuration


```python
epochs = 1000
opt = optax.adamw(1e-3, weight_decay=0.1)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
```


```python
@eqx.filter_jit
def train_step(model, opt_state, batch):
    x = batch
    def nll(model):
        _, logp_x = eqx.filter_vmap(model.train)(x)
        return -jnp.mean(logp_x)

    loss, grads = eqx.filter_value_and_grad(nll)(model)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss 

@eqx.filter_jit
def eval(model, key):
    keys = jr.split(key, 1000)
    x_pred_samples = jax.vmap(model.sample)(keys)
    x_pred_mu = jnp.mean(x_pred_samples, axis=0)
    x_pred_cov = jnp.cov(x_pred_samples, rowvar=False)
    mu_loss, std_loss = jnp.linalg.norm(x_pred_mu - x_true_mu) / jnp.linalg.norm(x_true_mu), jnp.linalg.norm(x_pred_cov - x_true_cov)/jnp.linalg.norm(x_true_cov)
    return mu_loss, std_loss
```

### Loop


```python
mu_loss, std_loss = [],[]
for epoch in range(epochs):
    key, epoch_key = jr.split(key)
    model, opt_state, loss = train_step(model, opt_state, x_true_samples)
    mu_loss, std_loss = eval(model, epoch_key)
    if epoch % 50 == 0:
        print(f'{epoch=}, nll: {loss.item():.5f}, mu_loss: {mu_loss.item():.5f}, std_loss: {std_loss.item():.5f}') 
    
```

    (4,)
    epoch=0, nll: 14.51849, mu_loss: 1.16346, std_loss: 0.86170
    epoch=50, nll: 6.67654, mu_loss: 0.91126, std_loss: 0.62241
    epoch=100, nll: 5.56207, mu_loss: 0.63072, std_loss: 0.38704
    epoch=150, nll: 5.06523, mu_loss: 0.34559, std_loss: 0.23343
    epoch=200, nll: 4.71705, mu_loss: 0.21382, std_loss: 0.18625
    epoch=250, nll: 4.42771, mu_loss: 0.08188, std_loss: 0.15012
    epoch=300, nll: 4.21064, mu_loss: 0.05217, std_loss: 0.34919
    epoch=350, nll: 4.06594, mu_loss: 0.04972, std_loss: 0.47907
    epoch=400, nll: 3.97155, mu_loss: 0.02404, std_loss: 0.72304
    epoch=450, nll: 3.91002, mu_loss: 0.04324, std_loss: 0.74194
    epoch=500, nll: 3.86787, mu_loss: 0.01797, std_loss: 0.71651
    epoch=550, nll: 3.83281, mu_loss: 0.02633, std_loss: 0.78955
    epoch=600, nll: 3.79404, mu_loss: 0.09435, std_loss: 0.89187
    epoch=650, nll: 3.74345, mu_loss: 0.07691, std_loss: 0.76098
    epoch=700, nll: 3.67931, mu_loss: 0.07551, std_loss: 0.64115
    epoch=750, nll: 3.60887, mu_loss: 0.03644, std_loss: 0.61345
    epoch=800, nll: 3.54310, mu_loss: 0.02968, std_loss: 0.49329
    epoch=850, nll: 3.48873, mu_loss: 0.10529, std_loss: 0.27902
    epoch=900, nll: 3.44688, mu_loss: 0.06321, std_loss: 0.34919
    epoch=950, nll: 3.41591, mu_loss: 0.00947, std_loss: 0.20908



```python

```
