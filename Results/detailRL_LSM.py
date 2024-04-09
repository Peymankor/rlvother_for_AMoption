
from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("..")

from random import randrange

from helper import continuation_curve_lsm, put_option_exercise_boundary
from RL_policy import fitted_lspi_put_option
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights

from LSM_policy import OptimalExerciseLSM
from numpy.polynomial.laguerre import lagval


from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal

## Import packages for Bin

from bintree_policy import bin_tree_price

## Import packages for LSM

from LSM_policy import  lsm_price

## Import packages for RL

from RL_policy import RL_GBM_Policy

#############################################################
#############################################################


# Example of Longstaff og Schwartz

spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 1.0
rate_val: float = 0.06
vol_val: float = 0.2
num_scoring_paths: int = 10000
num_steps_scoring: int = 50


#num_scoring_paths_val: int = 5000
num_steps_scoring: int = 50
num_steps_lspi: int = 15
num_training_paths_lspi_val: int = 20000
spot_price_frac_lspi: float = 0.0000000000000000000001
training_iters_lspi_val: int = 8

#num_steps_value_lsm = 50 
#num_paths_train_lsm = 10000
K_value = 40
k_value = 4
#num_paths_test_value_lsm = 10000

    #random.seed(100)
    #np.random.seed(100)
############# RL ###################################################
####################################################################

flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        num_paths=num_training_paths_lspi_val,
        spot_price=spot_price_val,
        spot_price_frac=spot_price_frac_lspi,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        training_iters=training_iters_lspi_val
    )

print("Fitted LSPI Model")
flspi
x
# lspi_x, lspi_y = put_option_exercise_boundary(
#         func=flspi,
#         expiry=expiry_val,
#         num_steps=num_steps_lspi,
#         strike=strike_val
#     )

############# LSM ###################################################
####################################################################

S0_table1 = 36
sd_table1 = 0.2
T_table1 = 1


num_scoring_paths_val = 10000

def payoff_func(_: float, s: float) -> float:
            return max(K_value - s, 0.)

Bin_list =[]
RL_list = []

for repitation in range(10):

    #price_bin = bin_tree_price(spot_price_val=S0_table1, strike_val=strike_val,
    #                                     expiry_val=T_table1, rate_val=rate_val, vol_val=sd_table1)
                     

    price_rl,_ = RL_GBM_Policy(spot_price_val=S0_table1, strike_val=strike_val, expiry_val=T_table1,
                      rate_val=rate_val, vol_val=sd_table1, num_scoring_paths=num_scoring_paths_val,
                      num_steps_scoring=num_steps_scoring, num_steps_lspi=num_steps_lspi,
                      num_training_paths_lspi=num_training_paths_lspi_val,spot_price_frac_lspi=spot_price_frac_lspi,
                      training_iters_lspi=training_iters_lspi_val)

    print("%20.3f"%(price_rl))

    #Bin_list.append(price_bin)
    RL_list.append(price_rl)

print(RL_list)
print(np.mean(RL_list))
#print("%20.3f %20.3f"%(price_bin, price_rl))