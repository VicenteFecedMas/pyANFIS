---
title: "Universe"
#type: "chapter"
weight: 10
chapter: true
---

```python
CLASS    pyanfis.functions.Universe(name, range, functions)
```

## Parameters

| name | type | description |
|------|------|-------------|
| **name** | str | name of the universe |
| **range** | tuple | range of the universe, from min to max where min < max |
| **functions** | dict | dict with names of functions and properties of functions |

In regards to the **functions** parameter, you should input a dictionary where each key is the name of the function and each value is a dict that indicates its type and and its parameters:
```json
{
    "Small": {
        "type": "LinearZ",
        "parameters": {
            "foot": 10,
            "shoulder": 0
        }
    },
    "Big": {
        "type": "LinearS",
        "parameters": {
            "shoulder": 10,
            "foot": 0
        }
    }
}
```

## Example

The first step is to import torch and the universe using:
```python
import torch

from pyanfis.functions import Universe
```

A universe will accept: a name, a range where the universe will be evaluated and a dictionary of functions that will comprise the different linguistical variables. It is easier to embed all the parameters into a dictionary and feed them into the system:
```python
params = {
   "name": "Dummy_Universe",
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
}
```

The parameters are passed to the universe using the two asterisks in front of the the dictionary:
```python
universe = Universe(**params)
```

Incidentally, you can also pass the parameters into the universe as:
```python
universe = Universe(
    name = "Dummy_Universe",
    range = (0, 3),
    functions = {
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
)
```

It is up to you how to do it, but the first option is prefered, as it will allow us to save all the parameters into a json that is separate from the main code, to help us abstract the parameter setting phase from the model instancing and training.

Next we need to create the input tensor, in this case must have 3 dimensions, as by design a universe will only accept batched tensors.

```python
x = torch.linspace(0, 3, 9).unsqueeze(0).unsqueeze(-1)
f_x = universe(x)
```

The input tensor ```x``` will be:
```python
tensor([[[0.0000],
         [0.3750],
         [0.7500],
         [1.1250],
         [1.5000],
         [1.8750],
         [2.2500],
         [2.6250],
         [3.0000]]])
```

And the output tensor ```f_x``` will be:
```python
tensor([[[1.0000, 0.3247, 0.0000],
         [0.8125, 0.5311, 0.0000],
         [0.6250, 0.7548, 0.0000],
         [0.4375, 0.9321, 0.0625],
         [0.2500, 1.0000, 0.2500],
         [0.0625, 0.9321, 0.4375],
         [0.0000, 0.7548, 0.6250],
         [0.0000, 0.5311, 0.8125],
         [0.0000, 0.3247, 1.0000]]], grad_fn=<CatBackward0>)
```


As expected, each input has been parsed through each of the 3 functions of the universe to give back a tensor of dimension (1, 9, 3)

## Visualisation

To visualize a function we will import the function and the matplotlib module:
```python
import matplotlib.pyplot as plt
import torch

from pyanfis.functions import Universe
```

Before plotting it, we will need the x values given by ```x``` and the y value given by ```f_x```:
```python
universe = Universe(**params)
x = torch.linspace(0, 3, 9).unsqueeze(0).unsqueeze(-1)
f_x = universe(x)
```

To plot the image use:
```python
plt.style.use("classic")
plt.title(universe.name)
plt.xlabel("x")
plt.ylabel("f (x)")
plt.margins(y=0.05)
for i in f_x[0,:,:].T:
   plt.plot(x[0, :, 0].detach(), i.detach())
plt.show()
```

And the final plotted function will be:

![Universe](/universe.png)