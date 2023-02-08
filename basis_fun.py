import numpy as np


#########################

# Legendre Polynomial

def legendre_polynomials(S, k):

    u0 = np.ones(S.shape)
    x1 = S
    x2 = (3*S**2 -1 )/2
    x3 = (5*S**3-3*S)/2
    x4 = (35*S**4-30*S**2 +3)/8
    x5 = (63*S**5 - 70*S**3 + 15*S)/8

    X = [np.stack([u0,x1,x2], axis=1),
         np.stack([u0, x1, x2, x3], axis= 1),
         np.stack([u0, x1, x2, x3, x4], axis=1)]


    return X[k-2]


# the case for one price, startin from left to right
def laguerre_polynomials_ind(S, k):
    

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2]),
          np.stack([u0, x1, x2, x3]),
          np.stack([u0, x1, x2, x3, x4])]
    
    return X[k-2]

def laguerre_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
#    x1 = np.exp(-S/2)
#    x2 = np.exp(-S/2) * (1 - S)
#    x3 = np.exp(-S/2) * (1 - 2*S + S**2/2)
#    x4 = np.exp(-S/2) * (1 - 3*S + 3* S**2/2 - S**3/6)

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]