# Exercise set 2

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 2.1

In the first exercise, we are going to implement a logical **AND** gate using PyTorch's 
`nn.Linear`. In particular, if you instantiate an object from `nn.Linear`, e.g., via 
```python
import torch
import torch.nn as nn
f = nn.Linear(2,1)
```
`f` will implement a map $f: \mathbb{R}^2 \to \mathbb{R}, \mathbf{x} \mapsto \langle \mathbf{w},\mathbf{x}\rangle + b$, where $\mathbf{w} \in \mathbb{R}^2$ and $b \in \mathbb{R}$, i.e., an *affine map*.  $\mathbf{w}$ and $b$ are the parameters of this function. You can access $\mathbb{w}$ via 
`f.weight` and $b$ via `f.bias`. The question now is what values you need to use in $\mathbb{w}$ and $b$ such that for inputs $\mathbf{x}_1 = [0,0]^\top$, $\mathbf{x}_2 = [0,1]^\top$, $\mathbf{x}_3 = [1,0]^\top$, $\mathbf{x}_4 = [1,1]^\top$ the correct output for the **AND** gate is returned. Implement this functionality in a way so you can use `return (f(x)>0).int()` as a final return 
statement.

### Exercise 2.2

same as Exercise 2.1, but for an **OR** gate.

*As in the first exercise sheet, you can evaluate your solution via*

```bash
otter check submission_XXX.py -q t1 # for Exercise 2.1
otter check submission_XXX.py -q t2 # for Exercise 2.2
```
