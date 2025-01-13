---
title: "Antecedents"
date: 2019-09-20T04:20:36+04:30
weight: 1
---

```python
CLASS    pyanfis.antecedents.Antecedents(universes)
```

## Parameters
The antecedents will receive a dictionary that inside will have a set of [universes]({{% ref "universe" %}}), 
- name(***dict***) - dictionary with "Input 1", "Input 2", ... as keys and universes as values.

This class is used to define the range in which a variable is going to be defined in a fuzzy way, it is composed of several functions used to describe it. 

## Example
```python
import torch

from pyanfis.antecedents import Antecedents

params = {
    "Input 1":{
    "name": "Dummy_Universe_1",
    "range": (0, 3),
    "functions": {
        "Small": {
            "type": "LinearZ",
            "parameters": {
                "shoulder": 0,
                "foot": 2                   
                }
            },
        "Medium": {
            "type": "Gauss",
            "parameters": {
                "mean": 1.5,
                "std": 1
            }
        },
        "Big": {
            "type": "LinearS",
            "parameters": {
                "foot": 1,
                "shoulder": 3
            }
        }
        }
    },
    "Input 2":{
    "name": "Dummy_Universe_2",
    "range": (0, 6),
    "functions": {
        "Small": {
            "type": "LinearZ",
            "parameters": {
                "shoulder": 0,
                "foot": 4                   
                }
            },
        "Medium": {
            "type": "Gauss",
            "parameters": {
                "mean": 3,
                "std": 2
            }
        },
        "Big": {
            "type": "LinearS",
            "parameters": {
                "foot": 2,
                "shoulder": 5
            }
        }
        }
    }
}
```

```python
antecedents = Antecedents(params)
x_1 = torch.linspace(0, 3, 9).unsqueeze(0).unsqueeze(-1)
x_2 = torch.linspace(0, 6, 9).unsqueeze(0).unsqueeze(-1)
x = torch.cat((x_1, x_2), dim=2)
f_x = antecedents(x)
```


```python
tensor([[[0.0000, 0.0000],
         [0.3750, 0.7500],
         [0.7500, 1.5000],
         [1.1250, 2.2500],
         [1.5000, 3.0000],
         [1.8750, 3.7500],
         [2.2500, 4.5000],
         [2.6250, 5.2500],
         [3.0000, 6.0000]]])
```


```python
tensor([[[1.0000, 0.3247, 0.0000, 1.0000, 0.3247, 0.0000],
         [0.8125, 0.5311, 0.0000, 0.8125, 0.5311, 0.0000],
         [0.6250, 0.7548, 0.0000, 0.6250, 0.7548, 0.0000],
         [0.4375, 0.9321, 0.0625, 0.4375, 0.9321, 0.0833],
         [0.2500, 1.0000, 0.2500, 0.2500, 1.0000, 0.3333],
         [0.0625, 0.9321, 0.4375, 0.0625, 0.9321, 0.5833],
         [0.0000, 0.7548, 0.6250, 0.0000, 0.7548, 0.8333],
         [0.0000, 0.5311, 0.8125, 0.0000, 0.5311, 1.0000],
         [0.0000, 0.3247, 1.0000, 0.0000, 0.3247, 1.0000]]], grad_fn=<IndexPutBackward0>)
```

## Visualization

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=len(antecedents.universes), figsize=(15, 5))
axs = axs.flatten()

for ax, (universe, i) in zip(axs, zip(antecedents.universes.values(), x.T)):
    ax.set_title(universe.name)
    ax.set_xlabel("x")
    ax.set_ylabel("f (x)")
    ax.margins(y=0.05)
    i = i.unsqueeze(0)
    ax.plot(i[0, :, :].detach(), universe(i)[0, :, :].detach())

plt.tight_layout()
plt.show()
```

![aefe47d0-de07-442c-b620-c0086368cead](https://github.com/user-attachments/assets/d90d32c6-5e52-41a0-bb32-c59d54f0771a)
