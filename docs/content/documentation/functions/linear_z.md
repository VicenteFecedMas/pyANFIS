---
title: Linear Z
weight: 10
---
```python
CLASS    pyanfis.functions.LinearZ(shoulder, foot)
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

from pyanfis.functions import LinearZ
```

After importing the function, we will initialise it with a shoulder value of 1 and a foot value of 2:
```python
linear_z = LinearZ(shoulder = 1, foot = 2)
x = torch.linspace(0, 3, 9)
f_x = linear_z(x)
```

The input tensor ```x``` will be:
```python
tensor([0.0000, 0.3750, 0.7500, 1.1250, 1.5000, 1.8750, 2.2500, 2.6250, 3.0000])
```

And the output tensor ```f_x``` will be:
```python
tensor([1.0000, 0.8750, 0.5000, 0.1250, 0.0000, 0.0000, 0.0000], grad_fn=<MaximumBackward0>)
```

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import LinearZ
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
linear_z = LinearZ(shoulder = 1, foot = 2)
x = torch.linspace(0, 3, 9)
f_x = linear_z(x)
```

To plot the image use:
```
plt.style.use("classic")
plt.title("Linear Z")
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
plt.plot(x.detach(), f_x.detach())
plt.show()
```

And the final plotted function will be:

![Linear Z function](/linear_z.png)