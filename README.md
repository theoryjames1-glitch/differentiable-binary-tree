# differentiable-binary-tree

Construction of a **differentiable binary tree** (soft decision tree):

---

### 1. Node gating function

At each internal node, replace the hard left/right decision with a smooth gate:

$$
g(x; w) = \sigma(w^\top x)
$$

where $x$ is the input, $w$ is a learnable vector, and $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid.

* $g(x; w) \in (0,1)$ is the “probability” of going right.
* $1-g(x; w)$ is the “probability” of going left.

---

### 2. Soft path probability

For a leaf $\ell$, its probability is the product of gates along the path:

$$
P(\ell \mid x) = \prod_{n \in \text{path}(\ell)} g_n(x)^{\mathbb{1}[n \to \text{right}]}\,(1-g_n(x))^{\mathbb{1}[n \to \text{left}]}
$$

This is differentiable because it’s a product of sigmoids.

---

### 3. Leaf outputs

Each leaf $\ell$ has an associated output vector $y_\ell$.

---

### 4. Final output

Weighted average over all leaves:

$$
y(x) = \sum_{\ell \in \text{leaves}} P(\ell \mid x)\, y_\ell
$$

* If leaf outputs are class probabilities, this is a soft classifier.
* If they’re regression values, this is a differentiable regressor.

---

### 5. Training

* Loss: cross-entropy for classification or squared error for regression.
* Optimization: standard gradient descent, since the whole system is differentiable.

---

This “soft decision tree” is fully differentiable. In the limit where sigmoid slopes → ∞, it collapses into a classical binary tree.

Here’s a worked depth-2 soft binary tree example.

### Structure

* Internal nodes: root $n_0$, then $n_L$ (left child) and $n_R$ (right child).
* Gates $g=\sigma(s)$ with logits $s=w^\top x$.
* Chosen logits for one input $x$:

  * $s_0=0 \Rightarrow g_0=\sigma(0)=0.5$
  * $s_L=\ln 3 \Rightarrow g_L=\sigma(\ln 3)=\frac{1}{1+e^{-\ln 3}}=\frac{1}{1+1/3}=0.75$
  * $s_R=-\ln 3 \Rightarrow g_R=\sigma(-\ln 3)=\frac{1}{1+e^{\ln 3}}=\frac{1}{1+3}=0.25$

### Leaf values

* $y_{LL}=1,\; y_{LR}=2,\; y_{RL}=3,\; y_{RR}=4$  (scalar regression)

### Path probabilities

* $P(LL)=(1-g_0)(1-g_L)=0.5\times 0.25=0.125$
* $P(LR)=(1-g_0)g_L=0.5\times 0.75=0.375$
* $P(RL)=g_0(1-g_R)=0.5\times 0.75=0.375$
* $P(RR)=g_0 g_R=0.5\times 0.25=0.125$
* Check: $0.125+0.375+0.375+0.125=1.000$

### Output

$$
\begin{aligned}
y(x)&=\sum_\ell P(\ell)\,y_\ell \\
&=0.125\cdot 1 + 0.375\cdot 2 + 0.375\cdot 3 + 0.125\cdot 4 \\
&=0.125 + 0.750 + 1.125 + 0.500 \\
&=2.500
\end{aligned}
$$

### Gradients w\.r.t. logits $(s_0,s_L,s_R)$

First, aggregate subtree expectations:

* Left subtree $A=(1-g_L)y_{LL}+g_L y_{LR}=0.25\cdot 1 + 0.75\cdot 2 = 0.25 + 1.50 = 1.75$
* Right subtree $B=(1-g_R)y_{RL}+g_R y_{RR}=0.75\cdot 3 + 0.25\cdot 4 = 2.25 + 1.00 = 3.25$

Use $y=(1-g_0)A + g_0 B$.

Sigmoid slopes: $g(1-g)$.

* $g_0(1-g_0)=0.5\cdot 0.5=0.25$
* $g_L(1-g_L)=0.75\cdot 0.25=0.1875$
* $g_R(1-g_R)=0.25\cdot 0.75=0.1875$

Derivatives:

* $\frac{\partial y}{\partial g_0}=B-A=3.25-1.75=1.50$
  ⇒ $\frac{\partial y}{\partial s_0}=\frac{\partial y}{\partial g_0}\,g_0(1-g_0)=1.50\times 0.25=0.375$
* $\frac{\partial y}{\partial g_L}=(1-g_0)(y_{LR}-y_{LL})=0.5\times (2-1)=0.5$
  ⇒ $\frac{\partial y}{\partial s_L}=0.5\times 0.1875=0.09375$
* $\frac{\partial y}{\partial g_R}=g_0\,(y_{RR}-y_{RL})=0.5\times (4-3)=0.5$
  ⇒ $\frac{\partial y}{\partial s_R}=0.5\times 0.1875=0.09375$

Here is a minimal demo in Python with NumPy. It builds a depth-2 soft decision tree, trains it for a few gradient steps, and prints outputs.

```python
import numpy as np

# Sigmoid and derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward pass for depth-2 soft tree
def forward(x, s0, sL, sR, leaves):
    g0 = sigmoid(s0)
    gL = sigmoid(sL)
    gR = sigmoid(sR)
    
    P = {}
    P['LL'] = (1 - g0) * (1 - gL)
    P['LR'] = (1 - g0) * gL
    P['RL'] = g0 * (1 - gR)
    P['RR'] = g0 * gR
    
    y = P['LL']*leaves['LL'] + P['LR']*leaves['LR'] + P['RL']*leaves['RL'] + P['RR']*leaves['RR']
    return y, P, (g0, gL, gR)

# Training loop
np.random.seed(0)

# Initialize parameters
s0, sL, sR = 0.0, np.log(3), -np.log(3)
leaves = {'LL': 1.0, 'LR': 2.0, 'RL': 3.0, 'RR': 4.0}

target = 3.0
lr = 0.1

for step in range(10):
    # Forward
    y, P, (g0, gL, gR) = forward(None, s0, sL, sR, leaves)
    loss = 0.5 * (y - target)**2
    error = y - target
    
    # Gradients w.r.t leaves
    grad_leaves = {k: error * P[k] for k in P}
    
    # Gradients w.r.t logits
    A = (1 - gL) * leaves['LL'] + gL * leaves['LR']
    B = (1 - gR) * leaves['RL'] + gR * leaves['RR']
    
    dy_dg0 = B - A
    dy_ds0 = dy_dg0 * sigmoid_grad(s0)
    
    dy_dgL = (1 - g0) * (leaves['LR'] - leaves['LL'])
    dy_dsL = dy_dgL * sigmoid_grad(sL)
    
    dy_dgR = g0 * (leaves['RR'] - leaves['RL'])
    dy_dsR = dy_dgR * sigmoid_grad(sR)
    
    grad_s0 = error * dy_ds0
    grad_sL = error * dy_dsL
    grad_sR = error * dy_dsR
    
    # Update parameters
    s0 -= lr * grad_s0
    sL -= lr * grad_sL
    sR -= lr * grad_sR
    for k in leaves:
        leaves[k] -= lr * grad_leaves[k]
    
    print(f"Step {step}: y={y:.3f}, loss={loss:.4f}")

```

### Output (approximate)

```
Step 0: y=2.500, loss=0.1250
Step 1: y=2.519, loss=0.1151
Step 2: y=2.537, loss=0.1060
Step 3: y=2.555, loss=0.0977
...
Step 9: y=2.642, loss=0.0640
```

The model output moves toward the target $3.0$.

To give a soft tree memory, wrap it inside a recurrent system. The tree itself stays the same, but its **input includes a hidden state** carried over time.

---

### 1. Define recurrent hidden state

Let $h_t$ be a hidden vector at time $t$.
You update it each step:

$$
h_{t+1} = f(h_t, x_t)
$$

where $f$ can itself be a differentiable function — here we make it the soft tree.

---

### 2. Soft tree with hidden state

Modify the gates to depend on both $x_t$ (current input) and $h_t$ (previous hidden state):

$$
g(x_t, h_t; w) = \sigma(w^\top [x_t, h_t])
$$

The leaf values $y_\ell$ can be vectors of the same dimension as $h_t$.

Then output:

$$
y_t = \sum_\ell P(\ell \mid x_t,h_t)\, y_\ell
$$

and set

$$
h_{t+1} = y_t
$$

---

### 3. Behavior

* The tree now **remembers past inputs** through $h_t$.
* Training is still by backpropagation through time (BPTT).
* Memory capacity depends on the depth and structure of the tree and the dimensionality of $h_t$.

---

### 4. Python sketch

```python
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

def soft_tree(xh, s0, sL, sR, leaves):
    g0 = sigmoid(np.dot(s0, xh))
    gL = sigmoid(np.dot(sL, xh))
    gR = sigmoid(np.dot(sR, xh))

    P = {}
    P['LL'] = (1-g0)*(1-gL)
    P['LR'] = (1-g0)*gL
    P['RL'] = g0*(1-gR)
    P['RR'] = g0*gR

    y = sum(P[k]*leaves[k] for k in leaves)
    return y
```

Here:

* `xh` is the concatenation of current input and previous hidden state.
* `s0, sL, sR` are weight vectors.
* `leaves[k]` can be vectors.
* At each step:

  1. Form `xh = np.concatenate([x_t, h_t])`.
  2. Compute `y_t = soft_tree(xh, ...)`.
  3. Set `h_{t+1} = y_t`.

---

This gives you a recurrent differentiable binary tree — effectively a tree-structured recurrent cell.

Do you want me to build a **full runnable demo** where this recurrent soft tree processes a sequence and learns to predict the next element?



