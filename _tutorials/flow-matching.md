---
title: "Flow Matching"
order: 5
notebook: /files/tutorials/flow_matching.ipynb
excerpt: "Simulation-free vector-field training along conditional probability paths, implemented with JAX, Equinox, Optax, and Diffrax."
---

Flow matching is to say we want a model akin to a continuous normalizing flow, wherein we aim to learn a transformation of a base density with $x_0 \sim p_0$ to a target density $x_{\text{data}} = x_1 \sim p_1$ by way of a solution map to an initial value problem, *but* we want to avoid expensive simulation during training. 

To accomplish this, we first make the observation that any probability path, $p_t: \mathbb{R}^d \times [0,1] \rightarrow \mathbb{R}$ implies a vector field and vice-versa from the continuity equation:

$$ \frac{d}{dt} p_t(x) + \nabla \cdot p_t f_t (x) = 0 $$

Which means that if we can somehow prescribe the probability path we want from $p_0$ to $p_1$, we could then generate some resultant vector field and find an approximation to this vector field.

$$ \mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim U(0,1), x_t \sim p_t(x)} \left[||f_\theta(x_t,t) - f(x_t,t)||^2 \right] $$

Unfortunately to *prescribe* a probability path we would actually need to know what our data distribution was, and if we knew that then we wouldn't be fiddling with this flow matching business in the first place. But there is a work around, wherein we instead write out a conditional probability path, which only requires drawing paths between distributions focused on individual data samples and our base distribution. For instance,

$$ 
\begin{aligned}
p_t(x|x_1) &= \mathcal{N}(x|\mu(x_1,t), \sigma^2(x_1,t)\mathbb{I}) \\
&= \mathcal{N}(x| tx_1, (1-t)^2 \mathbb{I})
\end{aligned}
$$

Notice we remain Gaussian throughout the entirety of the path, and that the mean and variance become dead centered at the data sample if $t=1$. This choice of path above (line 2) is linear, and conveniently means that the random variable $x_t = tx_1 + (1-t)x_0 \sim p_t$. From here, it can be deduced that $\frac{d}{dt} x_t = f(x_t, t| x_1) = x_1 - x_0$. Thus we have an analytical form of the ground truth vector field we want to approximate, with the caveat that for this choice of path our base distribution needs to be the standard univariate normal.

$$ \mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t\sim U(0,1), x_t \sim p_t(x|x_1), x_1 \sim p_1(x)} \left[f_\theta(x_t, t) - f(x_t, t; x_1)\right]
$$

And it turns out the optimal $\theta$ for this problem is the same as for the unconditional problem.


As far as the implementation is concerned, we need to (1) collect a a data sample $x_1$ and base sample $x_0$, (2) draw a straight line path between the two, as $x_t = t*x_1 + (1-t)*x_0$, (3) interpret the time derivative as $\frac{d}{dt} x_t = x_1 - x_0$ and set this as our ground truth vector field, for which we want our $f_\theta(x_t,t)$'s output to match over uniform random $t$. 


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

key = jr.PRNGKey(41)
```

### Choice of Parameterized Vector Field

Here we compose ConcatSquash layers from the [FFJORD](https://arxiv.org/pdf/1810.01367) [codebase](https://github.com/rtqichen/ffjord). 


```python
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, in_size, out_size, *, key):
        keys = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=keys[0])
        self.lin2 = eqx.nn.Linear(1, out_size, key=keys[1])
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=keys[2])

    def __call__(self, t, x):
        return self.lin1(x) * jax.nn.sigmoid(self.lin2(t)) + self.lin3(t)

### Compositional
class VectorField(eqx.Module):
    layers: List[eqx.Module]
    
    def __init__(self, base_layer, data_size, width_size, depth, *, key):
        layers = []
        keys = jr.split(key, depth+1)
        if depth == 0: layers.append(base_layer(in_size=data_size, out_size=data_size, key=keys[0]))
        else:
            layers.append(base_layer(in_size=width_size, out_size=width_size, key=keys[0]))
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

### Model 

The model has two functions, one which computes the loss between vector fields for training given a data sample x, the other which takes a random key to generate a sample the base distribution and thereafter solve the ode with our parameterized vector field forward in time to morph that sample to one which matches the likeness of the data. 


```python
class Flow(eqx.Module):
    base_dist: Callable
    Func: eqx.Module
    def __init__(self, *, vf, base_dist, key):
        self.Func = vf(key=key)
        self.base_dist = base_dist
     
    def train(self, x, key):
        x_1 = x
        x_0 = self.base_dist.rvs(key=key, shape=())
        ground_truth_dxdt = x_1 - x_0 ### x_dim

        key,_ = jr.split(key)
        t = jr.uniform(key)
        x_t = x_0 * (1 - t) + x_1 * t
        pred_dxdt = self.Func(t,x_t,args=None)
        return ((ground_truth_dxdt - pred_dxdt) ** 2).sum(axis=-1) 
        
    def sample(self, key):
        x = self.base_dist.rvs(key=key, shape=())
        term = diffrax.ODETerm(self.Func)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, 0, 1, 0.05, x)
        (x,) = sol.ys
        return x
```

### Problem setup

We aim to learn a correlated Gaussian distribution from a standard multivariate normal.


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
x_true_samples = jr.multivariate_normal(keys[2], mean=mu, cov=Sigma, shape=(n_samples,))
```


```python
class FixedStandardNormal(eqx.Module):
    dim: int
    def __init__(self, *, dim,):
        self.dim = dim
    def logpdf(self, x):
        return multivariate_normal.logpdf(x, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim))
    def rvs(self, key, shape):
        return jr.multivariate_normal(key, mean=jnp.zeros((self.dim,)), cov=jnp.eye(self.dim), shape=shape)

### could swap this to polynomial or kernel or whatever we want
base_layer = ConcatSquash
Func = partial(VectorField, base_layer=base_layer, data_size=x_dim, width_size=x_dim, depth=3)
model = Flow(vf=Func, 
             base_dist=TrainableStandardNormal(dim=x_dim, key=key),
             key=key)
```


```python
epochs = 10000
opt = optax.adamw(1e-3, weight_decay=0.1)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
```


```python
@eqx.filter_jit
def train_step(model, opt_state, batch, batch_key):
    x = batch
    keys = jr.split(batch_key, len(x))
    def vf_loss(model):
        loss = eqx.filter_vmap(model.train, in_axes=(eqx.if_array(0),0))(x, keys)
        return jnp.mean(loss)

    loss, grads = eqx.filter_value_and_grad(vf_loss)(model)
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
mu_losses, std_losses = [],[]
for epoch in range(10000):
    key, epoch_key = jr.split(key)
    model, opt_state, loss = train_step(model, opt_state, x_true_samples, epoch_key)
    mu_loss, std_loss = eval(model, epoch_key)
    mu_losses.append(mu_loss), std_losses.append(std_loss)
    if epoch % 100 == 0:
        print(f'{epoch=}, nll: {loss.item():.5f}, mu_loss: {mu_loss.item():.5f}, std_loss: {std_loss.item():.5f}') 
```

    epoch=0, nll: 6.78683, mu_loss: 1.12578, std_loss: 0.99546
    epoch=100, nll: 5.37291, mu_loss: 0.72612, std_loss: 0.99386
    epoch=200, nll: 4.80151, mu_loss: 0.40983, std_loss: 0.89230
    epoch=300, nll: 4.19211, mu_loss: 0.20704, std_loss: 0.78334
    epoch=400, nll: 3.93924, mu_loss: 0.12043, std_loss: 0.63566
    epoch=500, nll: 3.67770, mu_loss: 0.08436, std_loss: 0.59498
    epoch=600, nll: 3.27235, mu_loss: 0.07830, std_loss: 0.60900
    epoch=700, nll: 3.24856, mu_loss: 0.06482, std_loss: 0.60615
    epoch=800, nll: 3.13601, mu_loss: 0.05735, std_loss: 0.59457
    epoch=900, nll: 2.90206, mu_loss: 0.03743, std_loss: 0.61098
    epoch=1000, nll: 2.79929, mu_loss: 0.04993, std_loss: 0.47654
    epoch=1100, nll: 2.61687, mu_loss: 0.05145, std_loss: 0.35598
    epoch=1200, nll: 2.67581, mu_loss: 0.09829, std_loss: 0.31938
    epoch=1300, nll: 2.39045, mu_loss: 0.05367, std_loss: 0.26014
    epoch=1400, nll: 2.25182, mu_loss: 0.04693, std_loss: 0.23634
    epoch=1500, nll: 2.26016, mu_loss: 0.03203, std_loss: 0.19095
    epoch=1600, nll: 2.01537, mu_loss: 0.04188, std_loss: 0.19115
    epoch=1700, nll: 2.04963, mu_loss: 0.04987, std_loss: 0.17696
    epoch=1800, nll: 1.97661, mu_loss: 0.02995, std_loss: 0.13888
    epoch=1900, nll: 1.94153, mu_loss: 0.03836, std_loss: 0.17642
    epoch=2000, nll: 1.84767, mu_loss: 0.01916, std_loss: 0.16671
    epoch=2100, nll: 1.80324, mu_loss: 0.04174, std_loss: 0.12966
    epoch=2200, nll: 1.70289, mu_loss: 0.04857, std_loss: 0.09884
    epoch=2300, nll: 1.71861, mu_loss: 0.04569, std_loss: 0.10997
    epoch=2400, nll: 1.59341, mu_loss: 0.04076, std_loss: 0.08930
    epoch=2500, nll: 1.52193, mu_loss: 0.04462, std_loss: 0.09588
    epoch=2600, nll: 1.44928, mu_loss: 0.03043, std_loss: 0.11056
    epoch=2700, nll: 1.53462, mu_loss: 0.02776, std_loss: 0.09961
    epoch=2800, nll: 1.50857, mu_loss: 0.03400, std_loss: 0.11133
    epoch=2900, nll: 1.39257, mu_loss: 0.01962, std_loss: 0.13168
    epoch=3000, nll: 1.41946, mu_loss: 0.03906, std_loss: 0.10552
    epoch=3100, nll: 1.36475, mu_loss: 0.07450, std_loss: 0.09002
    epoch=3200, nll: 1.35982, mu_loss: 0.02161, std_loss: 0.10543
    epoch=3300, nll: 1.28038, mu_loss: 0.03088, std_loss: 0.10759
    epoch=3400, nll: 1.26522, mu_loss: 0.03687, std_loss: 0.10102
    epoch=3500, nll: 1.16870, mu_loss: 0.07326, std_loss: 0.11072
    epoch=3600, nll: 1.20292, mu_loss: 0.02479, std_loss: 0.11808
    epoch=3700, nll: 1.20770, mu_loss: 0.01830, std_loss: 0.11815
    epoch=3800, nll: 1.18423, mu_loss: 0.02377, std_loss: 0.11183
    epoch=3900, nll: 1.12776, mu_loss: 0.02471, std_loss: 0.09163
    epoch=4000, nll: 1.08054, mu_loss: 0.02732, std_loss: 0.10857
    epoch=4100, nll: 1.03960, mu_loss: 0.03468, std_loss: 0.06603
    epoch=4200, nll: 1.03789, mu_loss: 0.06000, std_loss: 0.05608
    epoch=4300, nll: 0.98228, mu_loss: 0.03826, std_loss: 0.10348
    epoch=4400, nll: 0.97961, mu_loss: 0.03825, std_loss: 0.07205
    epoch=4500, nll: 1.01662, mu_loss: 0.01337, std_loss: 0.07831
    epoch=4600, nll: 0.94699, mu_loss: 0.03421, std_loss: 0.05051
    epoch=4700, nll: 0.89433, mu_loss: 0.03123, std_loss: 0.11395
    epoch=4800, nll: 0.83366, mu_loss: 0.03154, std_loss: 0.11740
    epoch=4900, nll: 0.88572, mu_loss: 0.02262, std_loss: 0.08705
    epoch=5000, nll: 0.80878, mu_loss: 0.04125, std_loss: 0.05201
    epoch=5100, nll: 0.85631, mu_loss: 0.08513, std_loss: 0.04157
    epoch=5200, nll: 0.76074, mu_loss: 0.02326, std_loss: 0.13159
    epoch=5300, nll: 0.80029, mu_loss: 0.07382, std_loss: 0.06073
    epoch=5400, nll: 0.79764, mu_loss: 0.06022, std_loss: 0.08197
    epoch=5500, nll: 0.72835, mu_loss: 0.02604, std_loss: 0.10491
    epoch=5600, nll: 0.80264, mu_loss: 0.03551, std_loss: 0.07495
    epoch=5700, nll: 0.72545, mu_loss: 0.04371, std_loss: 0.06215
    epoch=5800, nll: 0.73001, mu_loss: 0.08180, std_loss: 0.09069
    epoch=5900, nll: 0.69819, mu_loss: 0.03391, std_loss: 0.08682
    epoch=6000, nll: 0.72048, mu_loss: 0.01521, std_loss: 0.12846
    epoch=6100, nll: 0.70075, mu_loss: 0.08808, std_loss: 0.06022
    epoch=6200, nll: 0.67897, mu_loss: 0.03365, std_loss: 0.06945
    epoch=6300, nll: 0.67439, mu_loss: 0.05962, std_loss: 0.07934
    epoch=6400, nll: 0.66992, mu_loss: 0.07839, std_loss: 0.07699
    epoch=6500, nll: 0.70602, mu_loss: 0.04007, std_loss: 0.07185
    epoch=6600, nll: 0.64036, mu_loss: 0.01346, std_loss: 0.09939
    epoch=6700, nll: 0.64344, mu_loss: 0.03871, std_loss: 0.13769
    epoch=6800, nll: 0.57825, mu_loss: 0.03842, std_loss: 0.10840
    epoch=6900, nll: 0.63950, mu_loss: 0.05337, std_loss: 0.09991
    epoch=7000, nll: 0.60266, mu_loss: 0.02685, std_loss: 0.08264



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[12], line 5
          3 key, epoch_key = jr.split(key)
          4 model, opt_state, loss = train_step(model, opt_state, x_true_samples, epoch_key)
    ----> 5 mu_loss, std_loss = eval(model, epoch_key)
          6 if epoch % 100 == 0:
          7     print(f'{epoch=}, nll: {loss.item():.5f}, mu_loss: {mu_loss.item():.5f}, std_loss: {std_loss.item():.5f}') 


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
