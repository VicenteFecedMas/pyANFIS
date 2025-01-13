---
title: Gauss
weight: 10
---
```python
CLASS    pyanfis.functions.Gauss(mean, std)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **mean** | int, float | mean of the gaussian function |
| **std** | int, float | standard deviation of the gaussian function |

## Example

The first step is to import torch and the function using:
```python
import torch

from pyanfis.functions import Gauss
```

After importing the function, we will initialise it with a mean value of 1.5 and a standard deviation value of 0.5:
```python
gauss = Gauss(mean = 1.5, std = 0.5)
x = torch.linspace(0, 3, 9)
f_x = gauss(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([0.0111, 0.0796, 0.3247, 0.7548, 1.0000, 0.7548, 0.3247, 0.0796, 0.0111], grad_fn=<ExpBackward0>)
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
gauss = Gauss(mean = 1.5, std = 0.5)
x = torch.linspace(0, 3, 9)
f_x = gauss(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Gauss")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Gauss function](/gauss.png)