---
title: Linear S
weight: 10
---
```python
CLASS    pyanfis.functions.LinearS(foot, shoulder)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **shoulder** | int, float | position of the shoulder of the function |
| **foot** | int, float | position of the foot of the function |

## Example

The first step is to import torch and the function using:
```python
import torch

from pyanfis.functions import LinearS
```

After importing the function, we will initialise it with a foot value of 1 and a shoulder value of 2:
```python
linear_s = LinearS(foot = 1, shoulder = 2)
x = torch.linspace(0, 3, 9)
f_x = linear_s(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([0.0000, 0.0000, 0.0000, 0.1250, 0.5000, 0.8750, 1.0000, 1.0000, 1.0000], grad_fn=<MinimumBackward0>)
```

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import LinearS
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
linear_s = LinearS(foot = 1, shoulder = 2)
x = torch.linspace(0, 3, 9)
f_x = linear_s(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Linear S")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Linear S function](/linear_s.png)