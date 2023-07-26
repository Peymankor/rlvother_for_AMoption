# Title of Your Paper

## Abstract

[Write a brief summary of your paper's main objectives and findings.]

## 1. Introduction

[Provide an introduction to your paper, explaining the background and context of the research.]

## 2. Literature Review

[Discuss relevant literature and previous work related to your research topic.]

## 3. Methodology

## LSM model

### The Pricing is based on set of sampling traces of the underlying prices

### Function approximation of the continuation value for in-the-money states

The LSM approach uses least squares to approximate the conditional expectaion function at $t_{K-1}, t_{K-2}, \cdots, t_1$:

- Unknown functional form:

$$F(\omega; t_{K-1})$$

- The unknown functional form can be represented as:

$$F(\omega; t_{K-1})=\sum_{j=0}^{\infty}a_jL_j(X)$$


In this work choice of basis function is set of Lageurre polynomials:

$$L_1(X)=1$$
$$L_2(X)= 1-X$$    
$$L_3(X) = 1 - 2X + X^2/2$$
$$L_4(X) = 1 - 3X + 3X^2/2 - X^3/6$$ 
$$L_5(X) = 1 - 4X + 3X^2 - 2X^3/3 + X^4/24$$

- In the python code unknown function can be represented as:

$$F(\omega; t_{K-1})=\sum_{j=0}^{5}a_jL_j(X)$$


In this work we focus only in-the-money paths in the estimation since the exercise decision is only relevant when the option is in the money.

### Backward -recursive determination of early exercise states

[Explain the methods and techniques you used to conduct your research.]

## 4. Results

[Present the findings and results of your research.]

### A) Bin, RL and LSM Option Valuation 

#### Part I) Example of Geometric Brownian Motion (GBM) - Constant Volatility Value Model

|   $S$ |   $\sigma$ |   $T$ |   Closed form European |   Binomial Tree |   LSM |  RL |
|-----:|------:|----:|-------------:|--------------:|------------:|-----------:|
|   36 |   0.2 |   1 |        3.844 |         4.488 |       4.472 |      4.42  |
|   36 |   0.2 |   2 |        3.763 |         4.846 |       4.837 |      4.772 |
|   36 |   0.4 |   1 |        6.711 |         7.119 |       7.108 |      6.954 |
|   36 |   0.4 |   2 |        7.7   |         8.508 |       8.514 |      8.298 |
|   38 |   0.2 |   1 |        2.852 |         3.26  |       3.255 |      3.197 |
|   38 |   0.2 |   2 |        2.991 |         3.748 |       3.741 |      3.669 |
|   38 |   0.4 |   1 |        5.834 |         6.165 |       6.131 |      5.987 |
|   38 |   0.4 |   2 |        6.979 |         7.689 |       7.669 |      7.472 |
|   40 |   0.2 |   1 |        2.066 |         2.316 |       2.309 |      2.247 |
|   40 |   0.2 |   2 |        2.356 |         2.885 |       2.906 |      2.814 |
|   40 |   0.4 |   1 |        5.06  |         5.31  |       5.316 |      5.151 |
|   40 |   0.4 |   2 |        6.326 |         6.914 |       6.89  |      6.724 |
|   42 |   0.2 |   1 |        1.465 |         1.622 |       1.624 |      1.598 |
|   42 |   0.2 |   2 |        1.841 |         2.217 |       2.221 |      2.156 |
|   42 |   0.4 |   1 |        4.379 |         4.602 |       4.593 |      4.447 |
|   42 |   0.4 |   2 |        5.736 |         6.264 |       6.236 |      6.055 |
|   44 |   0.2 |   1 |        1.017 |         1.117 |       1.114 |      1.071 |
|   44 |   0.2 |   2 |        1.429 |         1.697 |       1.694 |      1.665 |
|   44 |   0.4 |   1 |        3.783 |         3.956 |       3.975 |      3.86  |
|   44 |   0.4 |   2 |        5.202 |         5.652 |       5.658 |      5.429 |

#### Part II) Example of GARCH (1,1) -Stochastic Volatility Value Model

da

### Analyzing the distribution of option value

### Exploring decision criteria boundaries

• - Visualizing the stopping time for each policy.
• - Examining policy differences under varying uncertainties.
• - Discussing the distinctions between RL and the standard Bellman equation

## 5. Discussion

[Interpret the results and provide a discussion about their implications.]

## 6. Conclusion

[Summarize the key points of your paper and draw conclusions.]

## 7. References

[List all the references cited in your paper using APA or MLA citation style.]
