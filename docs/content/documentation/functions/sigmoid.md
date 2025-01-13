---
title: Sigmoid
weight: 10
---
```python
CLASS    pyanfis.functions.Sigmoid(width, center)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **width** | int, float | width of the transition area of the sigmoid function |
| **center** | int, float | center of the sigmoid function |

## Example

The first step is to import torch and the function using:
```python
import torch

from pyanfis.functions import Sigmoid
```

After importing the function, we will initialise it with a width value of 1 and a center value of 1:
```python
sigmoid = Sigmoid(width = 1, center = 1.5)
x = torch.linspace(0, 3, 9)
f_x = sigmoid(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([0.1824, 0.2451, 0.3208, 0.4073, 0.5000, 0.5927, 0.6792, 0.7549, 0.8176], grad_fn=<MulBackward0>)
```

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import Sigmoid
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
sigmoid = Sigmoid(width = 1, center = 1.5)
x = torch.linspace(0, 3, 9)
f_x = sigmoid(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Sigmoid")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Sigmoid function](/sigmoid.png)