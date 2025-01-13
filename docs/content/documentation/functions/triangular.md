---
title: Triangular
weight: 10
---
```python
CLASS    pyanfis.functions.Triangular(left_foot, peak, right_foot)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **left_foot** | int, float | right place where the base of the triangular function will be located |
| **peak** | int, float | place where the peak of the triangular function will be located |
| **right_foot** | int, float | left place where the base of the triangular function will be located |

## Example

The first step is to import torch and the function using:
```python
import torch

from pyanfis.functions import Triangular
```

After importing the function, we will initialise it with a left foot value of 1, a peak value of 2 and a right foot value of 3:
```python
triangular = Triangular(left_foot = 1, peak = 2, right_foot = 3)
x = torch.linspace(0, 3, 9)
f_x = triangular(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([0.0000, 0.0000, 0.0000, 0.1250, 0.5000, 0.8750, 0.7500, 0.3750, 0.0000], grad_fn=<MaximumBackward0>)
```

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import Gauss
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
triangular = Triangular(left_foot = 1, peak = 2, right_foot = 3)
x = torch.linspace(0, 3, 9)
f_x = triangular(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Triangular")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Triangular function](/triangular.png)