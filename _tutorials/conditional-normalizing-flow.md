---
title: "Conditional Normalizing Flow"
order: 2
notebook: /files/tutorials/cond_normalizing_flow.ipynb
excerpt: "Conditional coupling layers and conditional base densities for learning Gaussian inverse-problem posteriors."
---

In this code we'll be building a conditional normalizing flow, which consequently uses a lot of the same concepts as our previous normalizing flow tutorial. The idea is that we can learn a conditional distribution via

$$\begin{align}
p_{X|Y}(x|y) &= p_{Z|Y}(z|y) \left| \det \nabla_x z \right| \\
            &= p_{Z|Y}(T_\phi(x,y)|y) \left| \det \nabla_x T_\phi(x,y) \right|
\end{align}$$

Which is to say that $T_\phi: X \times Y \rightarrow Z$. To parse through what this is doing as compared to a standard normalizing flow, we (1) make our transform a function of both the variable we want to condition on 

and the variable whose conditional distribution we want to deduce, $Y$ and $X$ respectively, and (2) we also make the base density conditioned on $Y$.

Practically this is helpful for a variety of different problem settings. For instance, if we wanted to learn a different distribution for a given image label in a larger dataset of images.





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
from layers import NormalizingFlow
from functools import partial

key = jr.PRNGKey(42)
```

### Problem Setup
The case we'll consider here is a inverse problem of estimating the posterior distribution $p(x|y)$ according to the forward model:
$$ y = Ax + b + \epsilon$$ 
with fixed $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m, \text{ noise };
\epsilon \sim \mathcal{N}(0,\Sigma), x \sim \mathcal{N}(\mu, \Lambda)$ 

Thus we'll need both samples of $x \sim p(x)$ and the corresponding conditional variable value $y$ under this forward model for training, which are both generated below. 


```python
m,n = 2,4
keys = jr.split(key, 7)
A = jr.uniform(keys[0], (m,n))
b = jr.uniform(keys[1], (m))

n_obs = 1000

### noise
# L_eps = jr.normal(keys[2], (int(m*(m+1)/2)))
# L_eps = jnp.zeros((m,m)).at[jnp.tril_indices(m)].set(L_eps)
# Sigma = L_eps @ L_eps.T ### valid covariance
Sigma = jnp.diag(jr.uniform(keys[2],(m,)) + 2)

eps = jr.multivariate_normal(keys[3], mean=jnp.zeros((m,)), cov=Sigma, shape=(n_obs,)) ### (n_ym)

### prior, x
L_x = jr.normal(keys[4], (int(n*(n+1)/2)))
L_x = jnp.zeros((n,n)).at[jnp.tril_indices(n)].set(L_x)
Lambda = L_x @ L_x.T
mu = jr.normal(keys[5], (n,))

x = jr.multivariate_normal(keys[6], mean=mu, cov=Lambda, shape=(n_obs,)) ### (n_obs, n)

### observable
y = (A @ x.T).T + b + eps ### (n_obs, m)

```

Here we can verify our model quantitatively as we can analytically derive the posterior as another Gaussian and compare the statistics of the samples predicted by the model to these true statistics. 

$p(x|y) = \mathcal{N}(\mu_p, \Lambda_p)$ with

$$\mu_p = \Lambda_p(\Lambda^{-1} \mu + A^T \Sigma^{-1} (y - b)), \Lambda_p  = (\Lambda^{-1} + A^T \Sigma^{-1} A)^{-1}$$



```python
Sig_post = jnp.linalg.inv(jnp.linalg.inv(Lambda) + A.T @ jnp.linalg.inv(Sigma) @ A)
mu_post = lambda y: Sig_post @ (jnp.linalg.inv(Lambda) @ mu +  A.T @ jnp.linalg.inv(Sigma) @ (y - b))
mu_post = jax.vmap(mu_post)(y) ### n_obs, n
```

### How to make the base density conditioned on Y? [Welling et al. 2019](https://arxiv.org/pdf/1912.00042)

Simple, make parameters of a parametric distribution a function of Y. 
$$ p_{Z|Y} = \mathcal{N}(m_\theta (y), S_\phi(y)) $$


```python
class ConditionalDiagonalGaussianBaseDensity(eqx.Module):
    mu: eqx.Module
    Sig: eqx.Module
    def __init__(self, *, y_dim, z_dim, mu_function, Sig_function, key):
        keys = jr.split(key)
        self.mu = mu_function(in_size=y_dim, out_size=z_dim, key=keys[0])
        self.Sig = Sig_function(in_size=y_dim, out_size=z_dim, key=keys[1])
    def logpdf(self, z, y):
        return multivariate_normal.logpdf(z, mean=self.mu(y), cov=jnp.diag(jax.nn.softplus(self.Sig(y))))
    def rvs(self, y, key, shape):
        return jr.multivariate_normal(key, mean=self.mu(y), cov=jnp.diag(jax.nn.softplus(self.Sig(y))), shape=shape)

### ... or not
class FixedStandardNormal(eqx.Module):
    dim: int
    def __init__(self, *, dim,):
        self.dim = dim
    def logpdf(self, z, y):
        return multivariate_normal.logpdf(z, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim))
    def rvs(self, y, key, shape):
        return jr.multivariate_normal(key, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim), shape=shape)
```

### How to make the Transformation a function of x and y? 

[Welling et al. 2019](https://arxiv.org/pdf/1912.00042) makes a conditional coupling layer,
where the parameterized scale and translation functions are functions of a partition of 
latent variable z and conditional variable y.

$$ \begin{align}
z_1 &= x_1 \\
z_2 &= x_2 \circ \exp(s_\theta(z_1, y)) + t(z_1, y)
\end{align}$$


```python
class ConditionalCouplingLayer(eqx.Module):
    s: eqx.Module
    t: eqx.Module
    mask: Tuple[jax.Array]
    n_mask: Tuple[jax.Array]
    def __init__(self, *, s_function, t_function, y_dim, mask, key):

        keys = jr.split(key,) 
        assert mask.dtype == jnp.bool
        in_size, out_size = int((~mask).sum()), int(mask.sum())
        in_size+=y_dim
        
        self.s = s_function(in_size=in_size, out_size=out_size, key=keys[0])
        self.t = t_function(in_size=in_size, out_size=out_size, key=keys[1])
        self.mask = jnp.nonzero(mask)
        self.n_mask = jnp.nonzero(~mask)
        
    # The forward function is the way to sample the model. It takes in the latent variable z 
    # and the conditioning variable y
    def forward(self, z, y):
        zy = jnp.concatenate((z[self.n_mask], y))
        s,t = self.s(zy), self.t(zy)
        x = z.at[self.mask].set((z[self.mask] - t) * jnp.exp(-s))
        return x

    # The inverse function returns both the inverse and the jacobian determinant, 
    # as we'll need both for the objective. 
    def inverse(self, x, y): 
        xy = jnp.concatenate((x[self.n_mask], y))
        s, t = self.s(xy), self.t(xy)
        log_jac_det = jnp.sum(s)
        z = x.at[self.mask].set(x[self.mask]*jnp.exp(s) + t)
        return z, log_jac_det 
```

### Compositional Model


```python
class ConditionalNormalizingFlow(eqx.Module):
    base_layers: List[eqx.Module]
    base_dist: eqx.Module
    
    def __init__(self, *, base_layers, base_dist, key):
        keys = jr.split(key, len(base_layers))
        self.base_layers = []
        for i,base_layer in enumerate(base_layers):
            self.base_layers.append(base_layer(key=keys[i])) 
        self.base_dist = base_dist
        
    def sample(self, key, y):
        z = self.base_dist.rvs(y, key, shape=())
        for layer in self.base_layers[::-1]:
            z = layer.forward(z, y)
        x = z
        return x    

    def train(self, x, y):
        log_jac_det=0.
        for layer in self.base_layers:
            x,log_jac_det_l = layer.inverse(x, y)
            log_jac_det+=log_jac_det_l
        z = x
        logp_x = log_jac_det + self.base_dist.logpdf(z, y)
        return z, logp_x

```


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

### Choose either MLP or orthogonal Polynomials for base layers : $s, t, m, S$


```python
x_dim,z_dim,y_dim = n,n,m
### RealNVP
s_function = partial(eqx.nn.MLP, depth=2, width_size=32, activation=jax.nn.tanh)
t_function = s_function

# s_function = partial(PolynomialExpansion, degree=10, total_degree=10)
# t_function = s_function

base_layer = partial(ConditionalCouplingLayer, y_dim=y_dim, s_function=s_function, t_function=t_function)

m_function = partial(eqx.nn.MLP, depth=1, width_size=16, activation=jax.nn.tanh)
S_function = partial(eqx.nn.MLP, depth=1, width_size=16, activation=jax.nn.tanh)
```

### Instatiate Base Density


```python
key,_ = jr.split(key)
base_dist = ConditionalDiagonalGaussianBaseDensity(y_dim=y_dim,
                                                z_dim=z_dim,
                                                mu_function=m_function,
                                                Sig_function=S_function,
                                                key=key)
# base_dist = FixedStandardNormal(dim=z_dim)
```

### Instatiate the Compositional Model


```python
depth = 4

mask = jnp.zeros((x_dim,)).at[:(x_dim//2)].set(1.).astype(bool)
base_layers = [partial(base_layer, mask=mask^bool(i%2)) for i in range(depth)] ### xor does mask flipping

### we're also training log_p_z_cond_x
model = ConditionalNormalizingFlow(base_layers=base_layers, base_dist=base_dist, key=key)
```

### Training/Optimizer Configuration


```python
epochs = 10000
opt = optax.adamw(1e-3, weight_decay=0.1)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
```


```python
@eqx.filter_jit
def train_step(model, opt_state, batch):
    x, y = batch
    def nll(model):
        ### T is a function of x i.e. x_cond_y and y
        _, logp_x = jax.vmap(lambda x,y: model.train(x, y))(x, y)
        return -jnp.mean(logp_x)
        
    loss, grads = eqx.filter_value_and_grad(nll)(model)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss 


######################################
def sample_conditional(model, yi, key):
    keys = jr.split(key, 1000)
    x_cond_yi_samples = jax.vmap(lambda key: model.sample(key,yi))(keys) ### n_samples, x_dim
    return x_cond_yi_samples


def compute_stats_from_samples(samples):
    mu = jnp.mean(samples, axis=0) ### x_dim
    cov = jnp.cov(samples, rowvar=False) ### x_dim,x_dim
    return mu, cov


@eqx.filter_jit
def eval(model, y, key):
    x_cond_y_samples = jax.vmap(lambda yi: sample_conditional(model, 
                                                              yi, key), 
                                out_axes=1)(y) ### n_samples, n_y, x_dim

    mu_y, Sig_y = jax.vmap(compute_stats_from_samples, in_axes=1)(x_cond_y_samples) ### n_y, x_dim; n_y, x_dim, x_dim
    Sig_y = Sig_y.reshape(len(mu_y), x_dim*x_dim)
    mu_err = jnp.linalg.norm(mu_y - mu_post, axis=1) / jnp.linalg.norm(mu_post, axis=1)
    Sig_err = jax.vmap(lambda Sig_y: jnp.linalg.norm(Sig_y - Sig_post.flatten()))(Sig_y)
    return mu_err.mean(), Sig_err.mean()

```

### Loop


```python

for epoch in range(10000):
    key, epoch_key = jr.split(key)
    
    model, opt_state, loss = train_step(model, opt_state, (x, y))
    mu_err, Sig_err = eval(model, y, key)
    if epoch % 100 == 0:
        print(f'{epoch=}, nll:{loss.item():.5f}, mu_err:{mu_err.item()*100:.5f}, Sig_err: {Sig_err.item()*100:.5f}')
```

    epoch=0, nll:15.37307, mu_err:93.60878, Sig_err: 287.80229
    epoch=100, nll:3.73449, mu_err:21.23260, Sig_err: 133.77948
    epoch=200, nll:3.26742, mu_err:15.72937, Sig_err: 86.15236
    epoch=300, nll:3.15428, mu_err:13.78292, Sig_err: 67.51139
    epoch=400, nll:3.07841, mu_err:12.76062, Sig_err: 70.79310
    epoch=500, nll:3.03802, mu_err:11.68807, Sig_err: 64.29676
    epoch=600, nll:2.99745, mu_err:11.01426, Sig_err: 67.44334
    epoch=700, nll:2.96672, mu_err:11.38635, Sig_err: 61.77131
    epoch=800, nll:2.94597, mu_err:10.93978, Sig_err: 63.36559
    epoch=900, nll:2.94684, mu_err:11.28118, Sig_err: 66.67178
    epoch=1000, nll:2.92042, mu_err:11.15027, Sig_err: 80.83217
    epoch=1100, nll:2.86504, mu_err:11.47389, Sig_err: 67.06093
    epoch=1200, nll:2.83900, mu_err:10.98977, Sig_err: 78.40238
    epoch=1300, nll:2.81837, mu_err:11.28858, Sig_err: 81.66454
    epoch=1400, nll:2.82269, mu_err:11.19601, Sig_err: 77.50223



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[13], line 5
          2 key, epoch_key = jr.split(key)
          4 model, opt_state, loss = train_step(model, opt_state, (x, y))
    ----> 5 mu_err, Sig_err = eval(model, y, key)
          6 if epoch % 100 == 0:
          7     print(f'{epoch=}, nll:{loss.item():.5f}, mu_err:{mu_err.item()*100:.5f}, Sig_err: {Sig_err.item()*100:.5f}')


        [... skipping hidden 1 frame]


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/equinox/_jit.py:271, in _call(jit_wrapper, is_lower, args, kwargs)
        266     # We need to include the explicit `isinstance(marker, jax.Array)` check due
        267     # to https://github.com/patrick-kidger/equinox/issues/988
        268     if not isinstance(marker, jax.core.Tracer) and isinstance(
        269         marker, jax.Array
        270     ):
    --> 271         marker.block_until_ready()
        272 except JaxRuntimeError as e:
        273     # Catch Equinox's runtime errors, and re-raise them with actually useful
        274     # information. (By default XlaRuntimeError produces a lot of terrifying
        275     # but useless information.)
        276     if last_error_info is not None and "_EquinoxRuntimeError: " in str(e):


    KeyboardInterrupt: 



```python

```


```python

```


```python

```
