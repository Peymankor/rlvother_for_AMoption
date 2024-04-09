
import sys
sys.path.append("..")
from rich import print, pretty
pretty.install()
import numpy as np
from price_model import BrentGARCHPrice_Light
from helper import garch_array_to_list
from typing import Callable, List, Tuple, Sequence
TrainingDataType = Tuple[int, float, float]
from typing import Callable, Dict, Sequence, Tuple, List
from numpy.polynomial.laguerre import lagval
from rl.function_approx import LinearFunctionApprox, Weights, FunctionApprox
from typing import Callable, List, Tuple, Sequence



class OptionPricingRL_GARCH:
    def __init__(self, spot_price: float, strike: float, expiry: float, num_time_steps: int, num_train_paths: int, rate: float):
        self.spot_price = spot_price
        self.strike = strike
        self.expiry = expiry
        self.num_time_steps = num_time_steps
        self.num_train_paths = num_train_paths
        self.rate = rate
        self.scoring_prices = None
        self.brent_garch_price_generator = BrentGARCHPrice_Light()

    def generate_garch_rl_training_data(self) -> List[TrainingDataType]:
        garch_prices = self.brent_garch_price_generator.generate_paths(
            num_time_steps=self.num_time_steps,
            num_path_numbers=self.num_train_paths,
            initial_price=self.spot_price,
            expiry_val=self.expiry
        )
        return garch_array_to_list(garch_prices)
    
    def generate_garch_rl_scoring_data(self, num_scoring_paths: int) -> np.ndarray:
        garch_prices = self.brent_garch_price_generator.generate_paths(
            num_time_steps=self.num_time_steps,
            num_path_numbers=num_scoring_paths,
            initial_price=self.spot_price,
            expiry_val=self.expiry
        )
        self.scoring_prices = garch_prices

        return garch_prices

    def fitted_lspi_put_option_GARCH(self, training_data: Sequence[TrainingDataType], training_iters: int) -> LinearFunctionApprox[Tuple[float, float]]:
        num_laguerre: int = 4
        epsilon: float = 1e-3
        ident: np.ndarray = np.eye(num_laguerre)
        features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
        features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * self.strike)) *
                      lagval(t_s[1] / self.strike, ident[i]))
                     for i in range(num_laguerre)]
        features += [
            (lambda t_s, i=i: np.exp(-t_s[0] / (2 * self.expiry)) * lagval(t_s[0] / self.expiry, ident[i]))
            for i in range(num_laguerre)
        ]
        dt: float = self.expiry / self.num_time_steps
        gamma: float = np.exp(-self.rate * dt)
        num_features: int = len(features)
        states: Sequence[Tuple[float, float]] = [(i * dt, s) for i, s, _ in training_data]
        next_states: Sequence[Tuple[float, float]] = \
            [((i + 1) * dt, s1) for i, _, s1 in training_data]
        feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                             for x in states])
        next_feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                                  for x in next_states])
        non_terminal: np.ndarray = np.array(
            [i < self.num_time_steps - 1 for i, _, _ in training_data]
        )
        exer: np.ndarray = np.array([max(self.strike - s1, 0)
                                     for _, s1 in next_states])
        wts: np.ndarray = np.zeros(num_features)
        for iteration_number in range(training_iters):
            a_inv: np.ndarray = np.eye(num_features) / epsilon
            b_vec: np.ndarray = np.zeros(num_features)
            cont: np.ndarray = np.dot(next_feature_vals, wts)
            cont_cond: np.ndarray = non_terminal * (cont > exer)
            for i in range(len(training_data)):
                phi1: np.ndarray = feature_vals[i]
                phi2: np.ndarray = phi1 - \
                    cont_cond[i] * gamma * next_feature_vals[i]
                temp: np.ndarray = a_inv.T.dot(phi2)
                a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
                b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
            wts = a_inv.dot(b_vec)
        return LinearFunctionApprox.create(
            feature_functions=features,
            weights=Weights.create(wts)
        )
    
    def calculate_final_day_price(self, num_scoring_paths: int):
        #self.test_prices = self.generate_prices(self.num_test_paths_f, self.spot_price)
        positive_indexes = (self.strike -self.scoring_prices[:, -1]) > 0
        ss = (self.strike - self.scoring_prices[:, -1][positive_indexes]) / np.exp(self.rate * self.expiry)
        final_day_price = np.sum(ss) / num_scoring_paths

        return final_day_price

    def option_price(self, scoring_data: np.ndarray, func: FunctionApprox[Tuple[float, float]]) -> Tuple[float, np.ndarray]:
        num_paths: int = scoring_data.shape[0]
        num_steps: int = scoring_data.shape[1] 
        #stoptime_rl: np.ndarray = np.ones(num_paths) * (num_steps + 1)
        stoptime_rl: np.ndarray = np.ones(num_paths) * 51
        prices: np.ndarray = np.zeros(num_paths)
        dt: float = self.expiry / num_steps
        for i, path in enumerate(scoring_data):
            step: int = 1
            while step <= num_steps:
                t: float = step * dt
                exercise_price: float = max(self.strike - path[step-1], 0)
                continue_price: float = func.evaluate([(t, path[step-1])])[0] if step < num_steps else 0.
                step += 1
                if (exercise_price >= continue_price) and (exercise_price > 0):
                    prices[i] = np.exp(-self.rate * t) * exercise_price
                    stoptime_rl[i] = step - 1
                    #stoptime_rl[i] = step - 1
                    step = num_steps + 1
        return np.average(prices), stoptime_rl
    
    # def class
# num_scoring_paths_f = 10000
# num_train_paths_val = 5000
    
# Test = OptionPricingRL_GARCH(spot_price=80, strike=80, expiry=1, 
#                               num_time_steps=50, num_train_paths=num_train_paths_val, 
#                               rate=0.06)
# Train_data = Test.generate_garch_rl_training_data()
# func_rl = Test.fitted_lspi_put_option_GARCH(Train_data, training_iters=3)
#scoring_data = Test.generate_garch_rl_scoring_data(num_scoring_paths=num_scoring_paths_f)
#final_day_price = Test.calculate_final_day_price(num_scoring_paths=num_scoring_paths_f)
#option_price_rl, stoptime_rl = Test.option_price(scoring_data, func_rl)

#print("Fitted LSPI Model")
#from helper import put_option_exercise_boundary

#lspi_x, lspi_y = put_option_exercise_boundary(
#        func=func_rl,
#        expiry=1,
#        num_steps=50,
#        strike=80
#    )

####################################################################
####################################################################


from rich import print, pretty
pretty.install()
from typing import Callable, List, Tuple, Sequence
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

#import sys
#sys.path.append("..")
#import helper
#from helper import continuation_curve_lsm
from RL_policy import fitted_lspi_put_option
from rl.function_approx import DNNApprox, LinearFunctionApprox, FunctionApprox, DNNSpec, AdamGradient, Weights
from numpy.polynomial.laguerre import lagval
from price_model import BrentGARCHPrice_Light
from LSM_policy_GARCH import OptimalExerciseLSM_GARCH
import numpy as np

class OptionPricingLSM_GARCH:
    def __init__(self, spot_price: float, strike: float, expiry: int, num_time_steps: int, num_train_paths: int, rate: float):
        self.spot_price = spot_price
        self.strike = strike
        self.expiry = expiry
        self.num_time_steps = num_time_steps
        self.num_train_paths = num_train_paths
        self.rate = rate
        self.k_value = 4
        self.garchclass_LSM = None
        self.training_prices = None
        self.lsm_policy_coef_beta = None
        self.num_test_paths_f = 100000
        self.test_prices = None
        self.option_price_lsm = None
        self.stopping_time_lsm = None
        self.final_day_price = None
        self.lsm_bound_x = None
        self.lsm_bound_y = None

    def payoff_func(self, _: float, s: float) -> float:
        return max(self.strike - s, 0.)

    def generate_prices(self, num_paths: int, initial_price: float):
        brent_garch_price_generator = BrentGARCHPrice_Light()
        return brent_garch_price_generator.generate_paths(
            num_time_steps=self.num_time_steps,
            num_path_numbers=num_paths,
            initial_price=initial_price,
            expiry_val=self.expiry
        )

    #def evaluate_model(self):
    #    self.test_prices = self._generate_prices(self.num_test_paths_f, self.spot_price)
        
    def calculate_final_day_price(self):
        self.test_prices = self.generate_prices(self.num_test_paths_f, self.spot_price)
        positive_indexes = (self.strike - self.test_prices[:, -1]) > 0
        ss = (self.strike - self.test_prices[:, -1][positive_indexes]) / np.exp(self.rate * self.expiry)
        self.final_day_price = np.sum(ss) / self.num_test_paths_f

        return self.final_day_price


    def calculate_continuation_curve(self):
        self.lsm_bound_x, self.lsm_bound_y = continuation_curve_lsm(
            left_price=60,
            right_price=500,
            deltaprice=0.1,
            lsm_policy_coef=self.lsm_policy_coef_beta,
            K_value=self.strike,
            expiry_val=self.expiry,
            num_steps_value_lsm=self.num_time_steps
        )

        return self.lsm_bound_x, self.lsm_bound_y

    def train_model(self):
        self.garchclass_LSM = OptimalExerciseLSM_GARCH(
            spot_price=self.spot_price,
            payoff=self.payoff_func,
            expiry=self.expiry,
            num_steps=self.num_time_steps,
            rate=self.rate
        )

        self.training_prices = self.generate_prices(self.num_train_paths, self.spot_price)

        self.lsm_policy_coef_beta = self.garchclass_LSM.train_LSM(
            training_data=self.training_prices,
            num_paths_train=self.num_train_paths,
            K=self.strike,
            k=self.k_value
        )

        return self.lsm_policy_coef_beta

    def evaluate_model(self):
        #self.test_prices = self._generate_prices(self.num_test_paths_f, self.spot_price)

        self.option_price_lsm, self.stopping_time_lsm = self.garchclass_LSM.option_price(
            scoring_data=self.test_prices,
            Beta_list=self.lsm_policy_coef_beta,
            k=self.k_value
        )

        return self.option_price_lsm, self.stopping_time_lsm
        
#     def print_results(self):
#         print("Option Price LSM:", self.option_price_lsm)
#         print("Stopping Time LSM:", self.stopping_time_lsm)
#         print("Final Day Price:", self.final_day_price)
#         print("LSM Bound X:", self.lsm_bound_x)
#         print("LSM Bound Y:", self.lsm_bound_y)


# spot_price_val = 80.0
# strike_val = 80.0
# expiry_val = 1
# num_time_steps_val_f = 50
# num_train_paths_f = 100000
# rate_val = 0.06


# option_pricing_lsm = OptionPricingLSM_GARCH(
#         spot_price=spot_price_val,
#         strike=strike_val,
#         expiry=expiry_val,
#         num_time_steps=num_time_steps_val_f,
#         num_train_paths=num_train_paths_f,
#         rate=rate_val
# )

# training_prices = option_pricing_lsm.generate_prices(num_paths=num_train_paths_f, 
#                                                     initial_price=spot_price_val)

# lsm_garch_coef = option_pricing_lsm.train_model()
# # select first 100 rows on the training prices
# #training_prices = training_prices[:100]
# for row in training_prices[:100]:
#     plt.plot(row)
# plt.show()

# from helper import continuation_curve_lsm
# import numpy as np


# lsm_bound_x, lsm_bound_y = continuation_curve_lsm(
#                     left_price=0,right_price=80, 
#                     deltaprice=0.01, 
#                     lsm_policy_coef=lsm_garch_coef, 
#                     K_value=80, expiry_val=1, 
#                     num_steps_value_lsm= 50
#     )

# #lsm_garch_coef


# #############################################################
# #############################################################

# import matplotlib.pyplot as plt


# plt.plot(lsm_bound_x, lsm_bound_y, color='blue', label='LSM Boundary')
# plt.plot(lspi_x, lspi_y, color='red', label='RL Boundary')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Boundary Comparison: LSM vs RL')
# plt.legend()
# plt.show()

##lsm_bound_x
#lsm_bound_y

#lspi_x
#lspi_y

###############################################################
###############################################################

