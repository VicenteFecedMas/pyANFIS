---
title: "Takagi-Sugeno"
date: 2019-09-20T04:20:36+04:30
weight: 1
---

The Takagi-Sugeno Fuzzy Model also known as Adaptive Neuro-Fuzzy Inference Systems (ANFIS) is a Type 3 Fuzzy Inference System, where the final output is the weighted average of the outputs of all the rules and the rule outputs are a linear combination of the input variables and a constant. The following is a description of the IF-THEN rules for a Takagi-Sugeno system with three inputs:

```plaintext
Rule 1: IF ..., THEN f1 = p1 x + q1 y + r1z + s1.
Rule 2: IF ...,THEN f2 = p2 x + q2 y + r2z + s2.
Rule 3: IF ..., THEN f3 = p3 x + q3 y + r3z + s3.
```

Where, the inputs in the crisp set are denoted by ×,y, z (as mentioned in Table 6); linguistic labels by Ai, Bi, Ci; consequent parameters by pi, qi, ri and the output fuzzy membership functions by f1, f2, f3.
Five layers of linked neurons make up the typical ANFIS design, as shown in Fig. 9, which is indicative of artificial neural networks with similar functionality. In the Fig. 9, w1,w2 and w3 represents the weights of the neurons and ,  and  represents the normalized weights of the neurons (Chopra et al., 2021).