##

# This is the code to implement option pricing using the GARCH model as the underlying asset.

##

from typing import Callable, Dict, Sequence, Tuple, List
from price_model import BrentGARCHPrice_Light
import numpy as np
from rl.function_approx import LinearFunctionApprox, Weights, FunctionApprox
from random import randrange
from numpy.polynomial.laguerre import lagval
from helper import garch_array_to_list
import matplotlib.pyplot as plt

from rich import print, pretty
pretty.install()


TrainingDataType = Tuple[int, float, float]

########################################################################

spot_price_val = 80.0
strike_val = 80.0
expiry_val = 1
num_time_steps_val_f = 50
num_train_paths_f = 10
rate_val = 0.06

class RL_GARCH_Policy:
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
        num_steps: int = scoring_data.shape[1] - 1
        stoptime_rl: np.ndarray = np.ones(num_paths) * (num_steps + 1)
        prices: np.ndarray = np.zeros(num_paths)
        dt: float = self.expiry / num_steps
        for i, path in enumerate(scoring_data):
            step: int = 0
            while step <= num_steps:
                t: float = step * dt
                exercise_price: float = max(self.strike - path[step], 0)
                continue_price: float = func.evaluate([(t, path[step])])[0] if step < num_steps else 0.
                step += 1
                if (exercise_price >= continue_price) and (exercise_price > 0):
                    prices[i] = np.exp(-self.rate * t) * exercise_price
                    stoptime_rl[i] = step - 1
                    step = num_steps + 1
        return np.average(prices), stoptime_rl



spot_price_val = 80.0
strike_val = 80.0
expiry_val = 1
num_time_steps_val_f = 50
num_train_paths_f = 10000
num_scoring_paths_f = 10000
rate_val = 0.06

RL_GARCH_Policy = RL_GARCH_Policy(
    spot_price=spot_price_val,
    strike=strike_val,
    expiry=expiry_val,
    num_time_steps=num_time_steps_val_f,
    num_train_paths=num_train_paths_f,
    rate=rate_val
)

#RL_GARCH_Policy.generate_garch_rl_training_data()

training_data = RL_GARCH_Policy.generate_garch_rl_training_data()

func_rl = RL_GARCH_Policy.fitted_lspi_put_option_GARCH(training_data, training_iters=4)

scoring_data = RL_GARCH_Policy.generate_garch_rl_scoring_data(num_scoring_paths=num_scoring_paths_f)

final_day_price_rl = RL_GARCH_Policy.calculate_final_day_price(num_scoring_paths=num_scoring_paths_f)

print("GARCH Final Day Price:", final_day_price_rl)

option_price_rl, stoptime_rl = RL_GARCH_Policy.option_price(scoring_data, func_rl)

print("GARCH RL Option Price:", option_price_rl)

print("GARCH Rl Stopping Time:", stoptime_rl)

plt.hist(stoptime_rl, bins=50)
plt.show()