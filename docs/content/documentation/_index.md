---
title: "Documentation"
date: 2019-09-20T04:20:36+04:30
weight: 1
---
System modelling based on conventional mathematical tools (e.g., differential equations) is not well suited for dealing with ill-defined and uncertain systems. By contrast, a fuzzy inference system employing fuzzy if-then rules can model the qualitative aspects of human knowledge and reason- ing processes without employing precise quantitative analyses. This fuzzy modeling or fuzzy identification, first explored systematically by [Takagi and Sugeno](https://dns2.asia.edu.tw/~ysho/YSHO-English/2000%20Engineering/PDF/IEE%20Tra%20Sys%20Man%20Cyb15,%20116.pdf), has found numerous practical applications in control prediction and inference. However, there are some basic aspects of this approach which are in need of better understanding. More specifically:

1. No standard methods exist for transforming human knowledge or experience into the rule base and database of a fuzzy inference system.
2. There is a need for effective methods for tuning the membership functions (MF’s) so as to minimize the output error measure or maximize performance index.

In this perspective, the aim of this library is to suggest a novel architecture called Adaptive-Network-based Fuzzy Inference System, or simply ANFIS, which can serve as a basis for constructing a set of fuzzy if-then rules with appropriate membership functions to generate the stipulated input-output pairs.

## Input parameters
The parameters  can be stored as as a json with all the parameters or as 3 different jsons, storing: antecedent parameters, rules parameters, consequent parameters.

## [Antecedents]({{% ref "antecedents" %}})
This will be the first layer of an ANFIS. Every $node_i$ in this layer is a square node with a node function

$$O_{i}^{1}=\mu_{A_{i}}(x)$$

where $x$ is the input to $node_i$ , and A, is the linguistic label (small , large, etc.) associated with this node function. In other words, $O_{i}^{1}$ is the membership function of $A_{i}$ and it specifies the degree to which the given $x$ satisfies the quantifier $A_{i}$. Parameters in this layer are referred to as premise parameters.

## [Rules]({{< ref "rules" >}})
This is the second layer of an ANFIS, where fuzzy if-then rules are applied. Every node in this layer is a circle node labeled $\prod$, which multiplies the incoming signals and sends the product out. Each node represents a single rule's firing strength.

## Normalisation
This will be the third layer of an ANFIS. Every node in this layer is a circle node labeled $N$. The ith node calculates the ratio of the ith rule’s firing strength to the sum of all rules’ firing strengths:

$$O^{3} = \frac{w_{i}}{\sum {w_{i}}}$$

## [Consequents]({{< ref "consequents" >}})
This is the fourth layer of an ANFIS. Each node in this layer is a square node that computes the contribution of each rule to the overall output.

## Output
This is the fifth and final layer of an ANFIS. Each node in this layer is a circle node labeled $\Sigma$ that computes the overall output as the summation of all incoming signals:

$$O^{5}=\sum_{i} \overline{w_{i}} \cdot f_i$$