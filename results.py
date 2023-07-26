


## Two Different Volatility LSM

####
# This is code to price the option value of first example of 
# LSM paper, with three different method, BIn Tree, LSM, LSPI
####

from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights

from random import randrange
from numpy.polynomial.laguerre import lagval

from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves


from LSM_policy import OptimalExerciseLSM
from basis_fun import laguerre_polynomials_ind

from rich import print, pretty
pretty.install()

#######

from threemethods_GBM_analysis import european_put_price,training_sim_data
from threemethods_GBM_analysis import fitted_lspi_put_option, scoring_sim_data
from threemethods_GBM_analysis import continuation_curve_lsm,continuation_curve,exercise_curve, put_option_exercise_boundary
from threemethods_GBM_analysis import option_price


def lsm_bound(spot_price_val, expiry_val, rate_val, 
              vol_val, num_steps_value_lsm, num_paths_train_lsm,
              k_value, K_value):
      
    def payoff_func(_: float, s: float) -> float:
            return max(K_value - s, 0.)


    lsmclass = OptimalExerciseLSM(spot_price=spot_price_val,
                 payoff=payoff_func, expiry=expiry_val,
                 rate=rate_val, vol=vol_val,
                 num_steps=num_steps_value_lsm)

    train_data = lsmclass.GBMprice_training(
    num_paths_train=num_paths_train_lsm, seed_random=False)

    lsm_policy_coef_beta = lsmclass.train_LSM(training_data=train_data,
                            num_paths_train=num_paths_train_lsm, 
                            K=K_value, k=k_value)


    lsm_bound_x, lsm_bound_y = continuation_curve_lsm(
                    left_price=20,right_price=40, 
                    deltaprice=0.1, 
                    lsm_policy_coef=lsm_policy_coef_beta,K_value=K_value,
                    expiry_val= expiry_val, num_steps_value_lsm=num_steps_value_lsm
    )

    return lsm_bound_x, lsm_bound_y
    
spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 1.0
rate_val: float = 0.06
vol_val: float = 0.2
num_steps_value_lsm = 50 
num_paths_train_lsm = 100000
K_value = 40
k_value = 4
num_paths_test_value_lsm = 10000


#XX, YY = lsm_bound(spot_price_val=spot_price_val, expiry_val=expiry_val,
#          rate_val=rate_val, vol_val=vol_val, num_steps_value_lsm=num_steps_value_lsm,
#          num_paths_train_lsm=num_paths_train_lsm,
#          k_value=k_value, K_value=K_value)


vol_val1 = 0.2

vol_val2 = 0.4

lsm_bound_x_vol1, lsm_bound_y_vol1 = lsm_bound(spot_price_val=spot_price_val, expiry_val=expiry_val,
          rate_val=rate_val, vol_val=vol_val1, num_steps_value_lsm=num_steps_value_lsm,
          num_paths_train_lsm=num_paths_train_lsm,
          k_value=k_value, K_value=K_value)

lsm_bound_x_vol2, lsm_bound_y_vol2 = lsm_bound(spot_price_val=spot_price_val, expiry_val=expiry_val,
          rate_val=rate_val, vol_val=vol_val2, num_steps_value_lsm=num_steps_value_lsm,
          num_paths_train_lsm=num_paths_train_lsm,
          k_value=k_value, K_value=K_value)



plot_list_of_curves(
        list_of_x_vals=[lsm_bound_x_vol1, lsm_bound_x_vol2],
        list_of_y_vals=[lsm_bound_y_vol1, lsm_bound_y_vol2],
        list_of_colors=["g", "b"],
        list_of_curve_labels=["LSM: GBM with Volatility = {}".format(vol_val1), 
                                "LSM: GBM with Volatility = {}".format(vol_val2)],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, Binary Tree and LSM Exercise Boundaries"
    )

#############################################
#############################################

def rl_bound(expiry_val, strike_val, num_steps_lspi, spot_price_val, 
            num_training_paths_lspi, spot_price_frac_lspi, 
            training_iters_lspi, rate_val, vol_val):
      

    flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
    expiry=expiry_val,
    num_steps=num_steps_lspi,
    num_paths=num_training_paths_lspi,
    spot_price=spot_price_val,
    spot_price_frac=spot_price_frac_lspi,
    rate=rate_val,
    vol=vol_val,
    strike=strike_val,
    training_iters=training_iters_lspi
    )


    lspi_x, lspi_y = put_option_exercise_boundary(
        func=flspi,
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        strike=strike_val
    )

    return lspi_x, lspi_y


spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 1.0
rate_val: float = 0.06
vol_val: float = 0.2
num_scoring_paths: int = 10000
num_steps_scoring: int = 50

num_steps_lspi: int = 10
num_training_paths_lspi: int = 1000
spot_price_frac_lspi: float = 0.1
training_iters_lspi: int = 8

num_steps_value_lsm = 50 
num_paths_train_lsm = 100000
K_value = 40
k_value = 4
num_paths_test_value_lsm = 10000



vol_val1 = 0.2

vol_val2 = 0.1

lspi_bound_x_vol1, lspi_bound_y_vol1 = rl_bound(expiry_val=expiry_val, 
                                               strike_val=strike_val, 
                                               num_steps_lspi=num_steps_lspi, 
                                               spot_price_val=spot_price_val, 
            num_training_paths_lspi=num_training_paths_lspi, 
            spot_price_frac_lspi= spot_price_frac_lspi, 
            training_iters_lspi=training_iters_lspi, rate_val=rate_val, 
            vol_val=vol_val1)


lspi_bound_x_vol2, lspi_bound_y_vol2 = rl_bound(expiry_val=expiry_val, 
                                               strike_val=strike_val, 
                                               num_steps_lspi=num_steps_lspi, 
                                               spot_price_val=spot_price_val, 
            num_training_paths_lspi=num_training_paths_lspi, 
            spot_price_frac_lspi= spot_price_frac_lspi, 
            training_iters_lspi=training_iters_lspi, rate_val=rate_val, 
            vol_val=vol_val2)

lspi_bound_x_vol1
lsm_bound_y_vol1

lspi_bound_x_vol2
lsm_bound_y_vol2



plot_list_of_curves(
        list_of_x_vals=[lspi_bound_x_vol1, lspi_bound_x_vol2],
        list_of_y_vals=[lspi_bound_y_vol2, lspi_bound_y_vol2],
        list_of_colors=["g", "b"],
        list_of_curve_labels=["RL: GBM with Volatility = {}".format(vol_val1), 
                                "RL: GBM with Volatility = {}".format(vol_val2)],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI Exercise Boundaries"
    )
