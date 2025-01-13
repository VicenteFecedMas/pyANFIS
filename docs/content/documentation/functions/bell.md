---
title: Bell
weight: 10
---
```python
CLASS    pyanfis.functions.Bell(width, shape, center)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **width** | int, float | width of the bell function |
| **shape** | int, float | shape of the transition area of the bell function |
| **center** | int, float | center of the bell function |

## Example

The first step is to import torch and the function using:
```python
import torch

from pyanfis.functions import Bell
```

After importing the function, we will initialise it with a width value of 1, a shape value of 0.5 and a center value of 1.5:
```python
bell = Bell(width = 1, shape = 0.5, center = 1.5)
x = torch.linspace(0, 3, 9)
f_x = bell(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([0.4000, 0.4706, 0.5714, 0.7273, 1.0000, 0.7273, 0.5714, 0.4706, 0.4000], grad_fn=<MulBackward0>)
```

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import Bell
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
bell = Bell(width = 1, shape = 0.5, center = 1.5)
x = torch.linspace(0, 3, 9)
f_x = bell(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Bell")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Bell function](/bell.png)