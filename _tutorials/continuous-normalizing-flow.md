---
title: "Continuous Normalizing Flow"
order: 3
notebook: /files/tutorials/cont_normalizing_flow.ipynb
excerpt: "Continuous-time density transformation with Diffrax, Equinox vector fields, and the instantaneous change-of-variables formula."
---

This model transforms a random variable via solving an ODE with a parameterized vector field. 

$$ \frac{d}{d t} x_t = F_\theta(x_t, t)$$

with $F_\theta: \mathbb{R}^d \times \mathbb{R} \rightarrow \mathbb{R}^d$, $x(t=0) = x_0 = z$ (the latent variable) and $x(t=1) = x_1$ ideally matching our data distribution once the model is trained.

The log-density implied at any given time $t$ under this transformation follows the instantaneous change of variables formula, derived from the continuity equation:

$$ \frac{d}{d t}\log p_t (x_t) = -\nabla \cdot F(x_t, t) $$

Which is what we'll need as in a standard normalizing flow to train the model. Now in the same way that discrete normalizing flows need the corresponding $z$ under the inverse transformation of a data sample $x$ to evaluate the log-density, so too do we need it here. But now we have to **solve** for the corresponding $z = x_0$ from the initial condition $x_1$, in other words, backwards in time. 

$$ x_0 = x_1 + \int_1^0 F_\theta(x_t,t)dt $$

And simultaneously accumulate the change in log-density $\log p_0(x_0) - \log p_1(x_1)$ via the instananeous change of variables formula, plugging our estimate $x_0$ into $\log p_0(x_0)$ afterwards to get $\log p_1(x_1)$.

$$ 
\begin{align}
\log p_0(x_0) - \log p_1(x_1) &= \int_1^0 -\nabla \cdot F_\theta(x_t,t)dt \\
\log p_1(x_1) &= \log p_0(x_0) + \int_1^0 \nabla \cdot F_\theta(x_t,t)dt
\end{align} $$

Which begets the following system of ODEs:
$$ \begin{bmatrix}
x_0 \\
\log p_1(x_1) - \log p_0(x_0)
\end{bmatrix} 
= \int_1^0 \begin{bmatrix}
F_\theta(x_t, t) \\ 
\nabla \cdot F_\theta(x_t,t)
\end{bmatrix} $$

With initial conditions as: 
$$
\begin{bmatrix}
x_1 \\
\log p_1(x_1) - \log p_1(x_1)
\end{bmatrix} = 
\begin{bmatrix} x_1 \\ 0 \end{bmatrix}$$

We then attempt to maximize $\log p_1(x_1)$ for all of our data samples, i.e. minimize
$\mathcal{L}(\theta) = \mathbb{E}_{x_1 \sim q_{\text{data}}}[-\log p_\theta(x_1)]$

### Why ODEs?

In discrete normalizing flows, naive jacobian determinant calculation of the parameterized transformation is $O(d^3)$ where $d$ is the data dimension, and to get it down to a more reasonable and scalable $O(d)$ one has to enforce constraints such as its jacobian being triangular, which by consequence constrains its expressivity.

Here, ignoring the added ODE solve, the analogous expensive calculation is the divergence of $F$, and its naive computational complexity is $O(d^2)$; the only requirement being that $F$ is and its first derivatives be lipschitz. We can reduce this to linear time as before without this requirement on $F$ changing, using Hutchinson's trace estimator. Thus we hope for more expressive models using this approach. Note lipschitz continuity can be enforced on neural networks in practice by choosing smooth lipschitz activations, e.g. $tanh$.


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
import diffrax
from abc import ABC, abstractmethod

key = jr.PRNGKey(42)
```

### Problem Setup

As usual, we aim to learn a correlated multivariate Gaussian. $x \sim N(\mu, LL^T)$ with $\mu, L \sim U(0,1)$.


```python
x_dim = 4
keys = jr.split(key, 3)

n_samples = 1000

n_tri = int(x_dim*(x_dim+1)/2)
L = jr.uniform(keys[0], (n_tri,))
L = jnp.zeros((x_dim,x_dim)).at[jnp.tril_indices(x_dim)].set(L)
Sigma = L @ L.T ### valid covariance

mu = jr.uniform(keys[1], (x_dim,))
x_true_mu, x_true_cov = mu, Sigma
x_true_samples = jr.multivariate_normal(keys[2], mean=mu, cov=Sigma, shape=(n_samples))
```

### Base Distribution


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
        return multivariate_normal.logpdf(x, mean=jnp.zeros((dim,)), cov=jnp.eye(x_dim))
    def rvs(self, key, shape):
        return jr.multivariate_normal(key, mean=jnp.zeros((dim,)), cov=jnp.eye(x_dim), shape=shape)

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

### Now Define Vector Field

Here we compose ConcatSquash layers from the [FFJORD](https://arxiv.org/pdf/1810.01367) [codebase](https://github.com/rtqichen/ffjord). The ODE library we will be using is called [Diffrax](https://docs.kidger.site/diffrax/) and was written by the same author as Equinox, thus it is quite compatible. In this framework the input to a vector field must be $t, x, \text{args}$ as below. 


```python
### Base layer
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, *, in_size, out_size, key):
        keys = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=keys[0])
        self.lin2 = eqx.nn.Linear(1, out_size, key=keys[1])
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=keys[2])

    def __call__(self, t, x):
        return self.lin1(x) * jax.nn.sigmoid(self.lin2(t)) + self.lin3(t)

### Compositional
class VectorField(eqx.Module):
    layers: List[eqx.Module]
    def __init__(self, *, base_layer, data_size, width_size, depth, key):
        layers = []
        keys = jr.split(key, depth+1)
        if depth == 0: layers.append(base_layer(in_size=data_size, out_size=data_size, key=keys[0]))
        else:
            layers.append(base_layer(in_size=data_size, out_size=width_size, key=keys[0]))
            [layers.append(base_layer(in_size=width_size, out_size=width_size, key=k)) for k in keys[1:-1]]
            layers.append(base_layer(in_size=width_size, out_size=data_size, key=keys[-1]))
        self.layers = layers
        
    def __call__(self, t, x, args):
        t = jnp.asarray(t)[None] ### make t shape (1,) vs scalar
        for layer in self.layers[:-1]:
            x = layer(t, x)
            x = jax.nn.tanh(x)
        x = self.layers[-1](t, x)
        return x

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 2
          1 ### Base layer
    ----> 2 class ConcatSquash(eqx.Module):
          3     lin1: eqx.nn.Linear
          4     lin2: eqx.nn.Linear


    NameError: name 'eqx' is not defined


### Computing the integral of the divergence in tandem

As noted in [FFJORD](https://arxiv.org/pdf/1810.01367), the naive divergence calculation essentially requires summing $d$ evaluations of derivates, where each evaluation requires $d$ operations, thus $O(d^2)$. We can use Hutchinson's trace estimator to reduce this cost to linear time via one vector-jacobian product $O(d)$ and another dot product $O(d)$:

$$ \begin{align}
\text{Trace}(A) &= \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\epsilon^\top A \epsilon \right] \\
 \nabla \cdot F = \text{Trace}(\nabla_x F) &= \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\epsilon^\top (\nabla_x F) \epsilon \right] 
 \end{align} $$
 
Indeed we can also use the same $\epsilon$ throughout the duration of the solve without introducing bias (i.e. changing this expectation) due to Fibini's theorem.

$$ \begin{align}
\log p(x_1) &= \log p(x_0) + \int_1^0 \nabla \cdot F_\theta(x(t),t)dt \\
\log p(x_1) &= \log p(x_0) + \int_1^0 \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\epsilon^\top (\nabla_x F_\theta) \epsilon \right] \\
\log p(x_1) &= \log p(x_0) + \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[ \int_1^0 \epsilon^\top (\nabla_x F_\theta) \epsilon \right] \\
\end{align}$$



```python
### func is the vector field as defined above. We use jax.vjp to return its value at x_t as f, 
### and the vector jacobian product function, which looks like lambda x: x @ (df/dx). We then use 
### this function to either compute the exact divergence or the approximate divergence via hutchinson's trace estimator. 

def approx_logp_wrapper(t, x, args):
    x, _ = x
    eps, func = args
    fn = lambda x: func(t, x, args=None)
    f, vjp_fn = jax.vjp(fn, x)
    (eps_dfdx,) = vjp_fn(eps) ### e^T @ \nabla_x F
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp

def exact_logp_wrapper(t, x, args):
    x, _ = x
    _, func = args
    fn = lambda x: func(t, x, args=None)
    f, vjp_fn = jax.vjp(fn, x)
    (size,) = x.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdx,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdx)
    return f, logp    
```

### Choices for the adjoint

i.e. computing gradients through the differential equation. The default in diffrax is a binomial online checkpointing scheme [diffrax.RecursiveCheckpointAdjoint](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.RecursiveCheckpointAdjoint), which is preferred because it computes more accurate gradients than solving the continuous adjoint equations while keeping memory requirements low. Nonetheless we can still use the continuous adjoint equations with [diffrax.BacksolveAdjoint]([diffrax.RecursiveCheckpointAdjoint](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.BacksolveAdjoint) if we want. Be sure to look at the other options available in diffrax's [api](https://docs.kidger.site/diffrax/api/adjoints).


```python
# (1) adjoint = diffrax.BacksolveAdjoint()
# (2) adjoint_controller = diffrax.PIDController(atol=1e-6, rtol=1e-3, norm=diffrax.adjoint_rms_seminorm)
# adjoint = diffrax.BacksolveAdjoint(stepsize_controller=adjoint_controller)

# (3)
adjoint = diffrax.RecursiveCheckpointAdjoint()
```

### Model which does the solve!


```python

class CNF(eqx.Module):
    Func: eqx.Module
    t0: float
    t1: float
    dt0: float
    ODETerm: diffrax.ODETerm
    solver: diffrax.AbstractSolver
    base_dist: Callable
    x_dim: int
    
    def __init__(self, *, x_dim, Func, t0, t1, dt0, exact_logp, base_dist, key):
        key,_ = jr.split(key)
        self.x_dim = x_dim
        self.Func = Func(key=key)
        self.t0, self.t1, self.dt0 = t0, t1, dt0
        if exact_logp: self.ODETerm = diffrax.ODETerm(exact_logp_wrapper)
        else: self.ODETerm = diffrax.ODETerm(approx_logp_wrapper)
            
        self.solver = diffrax.Tsit5()
        self.base_dist = base_dist

    def train(self, x, key):
        eps = jr.normal(key, x.shape)
        delta_log_likelihood = 0.0
        
        x = (x, delta_log_likelihood)
        sol = diffrax.diffeqsolve(
            self.ODETerm, 
            self.solver, 
            self.t1, 
            self.t0, 
            -self.dt0, 
            x, 
            args=(eps, self.Func),
            adjoint=adjoint,
        )
        (x,), (delta_log_likelihood,) = sol.ys
        logp_x = delta_log_likelihood + self.base_dist.logpdf(x)
        ### return x_0 and log p_1(x_1)
        return x, logp_x

    def sample(self, key):
        x = self.base_dist.rvs(key=key, shape=())
        term = diffrax.ODETerm(self.Func)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, x)
        (x,) = sol.ys
        return x
        
```

### Instantiate Model


```python
key,_ = jr.split(key)

### Could swap this ConcatSquash layer to use polynomials/kernels also. 
base_layer = ConcatSquash
Func = partial(VectorField, base_layer=base_layer, data_size=x_dim, width_size=x_dim, depth=3)
model = CNF(x_dim=x_dim, Func=Func, t0=0.0, t1=1.0, dt0=0.05, base_dist=base_dist, exact_logp=True, key=key)
```

### Training/Optimizer Configuration


```python
epochs = 10000
lr_schedule = optax.schedules.cosine_onecycle_schedule(epochs, peak_value=1e-3)
opt = optax.adamw(lr_schedule, weight_decay=0.1)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
```


```python
@eqx.filter_jit
def train_step(model, opt_state, batch, batch_key):
    x = batch
    keys = jr.split(batch_key, len(x))
    def nll(model):
        _, log_px = eqx.filter_vmap(model.train, in_axes=(eqx.if_array(0),0))(x, keys)
        return -jnp.mean(log_px)

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
for epoch in range(10000):
    key, epoch_key = jr.split(key)
    model, opt_state, loss = train_step(model, opt_state, x_true_samples, epoch_key)
    mu_loss, std_loss = eval(model, epoch_key)
    if epoch % 50 == 0:
        print(f'{epoch=}, nll: {loss.item():.5f}, mu_loss: {mu_loss.item():.5f}, std_loss: {std_loss.item():.5f}') 
```

    epoch=0, nll: 6.08125, mu_loss: 0.88349, std_loss: 0.74791
    epoch=50, nll: 6.06588, mu_loss: 0.83961, std_loss: 0.75919



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[35], line 4
          2 for epoch in range(10000):
          3     key, epoch_key = jr.split(key)
    ----> 4     model, opt_state, loss = train_step(model, opt_state, x_true_samples, epoch_key)
          5     mu_loss, std_loss = eval(model, epoch_key)
          6     if epoch % 50 == 0:


        [... skipping hidden 3 frame]


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/jax/_src/pjit.py:292, in _cpp_pjit.<locals>.cache_miss(*args, **kwargs)
        287 if config.no_tracing.value:
        288   raise RuntimeError(f"re-tracing function {jit_info.fun_sourceinfo} for "
        289                      "`jit`, but 'no_tracing' is set")
        291 (outs, out_flat, out_tree, args_flat, jaxpr, attrs_tracked, box_data,
    --> 292  executable, pgle_profiler) = _python_pjit_helper(fun, jit_info, *args, **kwargs)
        294 maybe_fastpath_data = _get_fastpath_data(
        295     executable, out_tree, args_flat, out_flat, attrs_tracked, box_data,
        296     jaxpr.effects, jaxpr.consts, jit_info.abstracted_axes, pgle_profiler)
        298 return outs, maybe_fastpath_data, _need_to_rebuild_with_fdo(pgle_profiler)


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/jax/_src/pjit.py:153, in _python_pjit_helper(fun, jit_info, *args, **kwargs)
        151   args_flat = map(core.full_lower, args_flat)
        152   core.check_eval_args(args_flat)
    --> 153   out_flat, compiled, profiler = _pjit_call_impl_python(*args_flat, **p.params)
        154 else:
        155   out_flat = pjit_p.bind(*args_flat, **p.params)


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/jax/_src/pjit.py:1877, in _pjit_call_impl_python(jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs, *args)
       1869     fingerprint = fingerprint.hex()
       1870   distributed_debug_log(("Running pjit'd function", name),
       1871                         ("in_shardings", in_shardings),
       1872                         ("out_shardings", out_shardings),
       (...)   1875                         ("abstract args", map(core.abstractify, args)),
       1876                         ("fingerprint", fingerprint))
    -> 1877 return compiled.unsafe_call(*args), compiled, pgle_profiler


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/jax/_src/profiler.py:354, in annotate_function.<locals>.wrapper(*args, **kwargs)
        351 @wraps(func)
        352 def wrapper(*args, **kwargs):
        353   with TraceAnnotation(name, **decorator_kwargs):
    --> 354     return func(*args, **kwargs)


    File ~/miniconda3/envs/mk/lib/python3.12/site-packages/jax/_src/interpreters/pxla.py:1297, in ExecuteReplicated.__call__(self, *args)
       1294 if (self.ordered_effects or self.has_unordered_effects
       1295     or self.has_host_callbacks):
       1296   input_bufs = self._add_tokens_to_inputs(input_bufs)
    -> 1297   results = self.xla_executable.execute_sharded(
       1298       input_bufs, with_tokens=True
       1299   )
       1301   result_token_bufs = results.disassemble_prefix_into_single_device_arrays(
       1302       len(self.ordered_effects))
       1303   sharded_runtime_token = results.consume_token()


    KeyboardInterrupt: 



```python

```


```python

```


```python

```
