
########Decision Boundary Curves##################

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


num_scoring_paths_val: int = 10000
num_steps_scoring: int = 50
num_steps_lspi: int = 10
num_training_paths_lspi_val: int = 2000
spot_price_frac_lspi: float = 0.000000000000000000000000001
training_iters_lspi_val: int = 4

num_steps_value_lsm = 50 
num_paths_train_lsm = 10000
K_value = 40
k_value = 4
num_paths_test_value_lsm = 10000

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

lspi_x, lspi_y = put_option_exercise_boundary(
        func=flspi,
        expiry=expiry_val,
        num_steps=num_steps_scoring,
        strike=strike_val
    )

############# LSM ###################################################
####################################################################


def payoff_func(_: float, s: float) -> float:
            return max(K_value - s, 0.)


lsmclass = OptimalExerciseLSM(spot_price=spot_price_val,
                 payoff=payoff_func, expiry=expiry_val,
                 rate=rate_val, vol=vol_val,
                 num_steps=num_steps_value_lsm)

train_data_v = lsmclass.GBMprice_training(num_paths_train=num_paths_train_lsm, seed_random=False)

lsm_policy_coef_beta = lsmclass.train_LSM(training_data=train_data_v,
                            num_paths_train=num_paths_train_lsm, 
                            K=K_value, k=k_value)
    
print("Fitted LSM Model")


lsm_bound_x, lsm_bound_y = continuation_curve_lsm(
                    left_price=20,right_price=40, 
                    deltaprice=0.1, 
                    lsm_policy_coef=lsm_policy_coef_beta, 
                    K_value=K_value, expiry_val=expiry_val, 
                    num_steps_value_lsm= num_steps_value_lsm
    )

import numpy as np
from basis_fun import laguerre_polynomials_ind

prices = np.arange(30,40,1)

prices

cp = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
                      lsm_policy_coef_beta[0]) for p in prices])
cp
K_value = 40
ep = np.array([max(K_value - p, 0) for p in prices])
prices
ep
cp
import matplotlib.pyplot as plt

plt.plot(prices, cp, label='Continuation Value')
plt.plot(prices, ep, label='Exercise Payoff')
plt.xlabel('Prices')
plt.ylabel('Value')
plt.title('Continuation Value vs Exercise Payoff')
plt.legend()
plt.show()

##lsm_bound_x
#lsm_bound_y
########################## Bin ######################################
#####################################################################

opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike_val - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=100
    )


print("Fitted Bin Model")

vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
bin_tree_price: float = vf_seq[0][NonTerminal(0)]
bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)



##################### Plotting #######################
####################################################

from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_line, labs
from plotnine import themes, element_text, theme

from plotnine.data import mtcars

############ I AM HERE ###############

X_price_ratio= np.concatenate((np.array(lspi_x), np.array(lsm_bound_x), np.array(bin_tree_x)))

Y_boundary_price = np.concatenate((np.array(lspi_y), np.array(lsm_bound_y), np.array(bin_tree_y)))


Z_group_method = np.concatenate((np.full(len(lspi_y), "Q-Learning"),
                                 np.full(len(lsm_bound_y), "LSM"),
                                  np.full(len(bin_tree_y), "BOPM")))


data = {"price value": X_price_ratio,
        "Boundary": Y_boundary_price,
         "Method": Z_group_method }

data_option = pd.DataFrame(data)


zz = (ggplot(data_option, aes("price value", "Boundary", color="factor(Method)"))+ 
      geom_line(size = 1) +
      labs(x="Time (t/expiry)",
           y = "Underlying Asset Value") +
           labs(color = 'Method') +
           theme(legend_position = "bottom") +
           theme(text = element_text(size = 10)))                    # All font sizes)
print(zz)

#zz.save("boundary_curve_new.png", dpi=600)