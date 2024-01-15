# pyANFIS

Welcome to pyANFIS! here you will be able to find a framework that will allow you to use **Fuzzy Logic** with usual pytorch layers.
This framework is based on [Jang's](https://www.researchgate.net/publication/3113825_ANFIS_Adaptive-Network-based_Fuzzy_Inference_System?enrichId=rgreq-15825cac70a3ae795992310484420cab-XXX&enrichSource=Y292ZXJQYWdlOzMxMTM4MjU7QVM6MTU5MDc1MDY1MTQ3MzkyQDE0MTQ5Mzc4NTk3MzI%3D&el=1_x_2&_esc=publicationCoverPdf) original paper, although it is going to implement several more things (listed below).

## 2024 Roadmap

- [x] Jang's Original ANFIS.
- [] Create documentation for each class and function.
- [] Create an instalation tutorial.
- [] Consequent parameters can be estimated using backpropagation.
- [] Type 1 (Tsukamoto) consequents can be used.
- [] Type 2 (Lee) consequents can be used.
- [] Antecedents can merge when 2 functions cover a similar area.
- [] Automatically not use rules when they are not relevant.
- [] Display what rules have been fired with a certain data.
- [] Create a method to substitute a trained ANFIS with a surface like in [Matlab](https://www.mathworks.com/help/fuzzy/genfis.html#d126e35957)