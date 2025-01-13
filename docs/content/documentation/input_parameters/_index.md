---
title: "Input parameters"
date: 2019-09-20T04:20:36+04:30
weight: 1
---

The model will need a certain kind of parameters to be able to run properly. The system will accept three main parameters: antecedents, rules and consequents. The system will accept accept a big quantity of parameters, so it is advised to store the parameters on a different json. In the following subsections it will be exposed the structure of each main parameter group.

## Antecedents
The [antecedents]({{% ref "antecedents" %}}) json will contain all the information related to the [universes]({{% ref "universe" %}}) present in the antecedents. Each main key in this dictionary will be used to indicate which input is being used (Input 1, 2, 3, ....). Each value related to a key will contain a dictionary with: 

- A name for the universe.
- A range for the universe (in the form of a list, that goes from min to max).
-  dictionary with 

```json
{
    "Input 1": {
        "name": "Num_1",
        "range": [0, 10],
        "functions": {
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
    }
}
```