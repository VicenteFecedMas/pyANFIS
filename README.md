# pyanfis

## Introduction
Welcome to pyanfis! here you will be able to find a project that will allow you to use **Fuzzy Logic** in conjunction with pytorch.
This framework is based on [Jang's](https://www.researchgate.net/publication/3113825_ANFIS_Adaptive-Network-based_Fuzzy_Inference_System?enrichId=rgreq-15825cac70a3ae795992310484420cab-XXX&enrichSource=Y292ZXJQYWdlOzMxMTM4MjU7QVM6MTU5MDc1MDY1MTQ3MzkyQDE0MTQ5Mzc4NTk3MzI%3D&el=1_x_2&_esc=publicationCoverPdf).

## Why should I use pyanfis?
You should use pyanfis if:

1. You aim to **handle non-linearities** between inputs and outputs. Unlike feed-forward neural networks, which might require a larger number of layers and neurons to capture complex non-linearities, ANFIS uses fuzzy logic to model these relationships more efficiently.

2. You want to **add Interpretability to your model**, as ANFIS systems provide a clear understanding of how inputs are transformed into outputs.

3. An ANFIS can achieve comparable performance to deep neural networks with **fewer training samples**.

4. An ANFIS model will allow you to **incorporate domain-specific knowledge into the model** through the definition of fuzzy rules and membership functions. 

5. If your models are prone to overfitting, an ANFIS and its fuzzy logic-based structure will **inherently imposes constraints on the model complexity**, which helps prevent overfitting.

## What problems can I solve with pyanfis?
Currently pyanfis has only been tested can be used to solve prediction problems and control problems. In future updates, it will be posible to use it in conjunction with convolutional layers to classify images or to substitude encoders/decoders in different applications.

## How can I install pyanfis?
You just need to use on your terminal:

```bash
pip install pyanfis
```
or
```bash
pip3 install pyanfis
```