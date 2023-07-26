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
#from rl.gen_utils.plot_funcs import plot_list_of_curves


##from LSM_policy import OptimalExerciseLSM
#from basis_fun import laguerre_polynomials_ind

from rich import print, pretty
pretty.install()

TrainingDataType = Tuple[int, float, float]


def european_put_price(
    spot_price: float,
    expiry: float,
    rate: float,
    vol: float,
    strike: float
) -> float:
    sigma_sqrt: float = vol * np.sqrt(expiry)
    d1: float = (np.log(spot_price / strike) +
                 (rate + vol ** 2 / 2.) * expiry) \
        / sigma_sqrt
    d2: float = d1 - sigma_sqrt
    return strike * np.exp(-rate * expiry) * norm.cdf(-d2) \
        - spot_price * norm.cdf(-d1)


def training_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float
) -> Sequence[TrainingDataType]:
    ret: List[TrainingDataType] = []
    dt: float = expiry / num_steps
    spot: float = spot_price
    vol2: float = vol * vol

    mean2: float = spot * spot
    var: float = mean2 * spot_price_frac * spot_price_frac
    log_mean: float = np.log(mean2 / np.sqrt(var + mean2))
    log_stdev: float = np.sqrt(np.log(var / mean2 + 1))

    for _ in range(num_paths):
        price: float = np.random.lognormal(log_mean, log_stdev)
        
        #print("Initial Starting Price", price )
        
        for step in range(num_steps):
            m: float = np.log(price) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            next_price: float = np.exp(np.random.normal(m, np.sqrt(v)))
            ret.append((step, price, next_price))
            price = next_price
    return ret

def fitted_lspi_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> LinearFunctionApprox[Tuple[float, float]]:

    num_laguerre: int = 4
    epsilon: float = 1e-3
    #print(epsilon)

    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    num_features: int = len(features)
    states: Sequence[Tuple[float, float]] = [(i * dt, s) for
                                             i, s, _ in training_data]
    next_states: Sequence[Tuple[float, float]] = \
        [((i + 1) * dt, s1) for i, _, s1 in training_data]
    feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                         for x in states])
    next_feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                              for x in next_states])
    non_terminal: np.ndarray = np.array(
        [i < num_steps - 1 for i, _, _ in training_data]
    )
    exer: np.ndarray = np.array([max(strike - s1, 0)
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
        #print(f"Iteration Number = {iteration_number:.3f}")

    return LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(wts)
    )


def scoring_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    rate: float,
    vol: float
) -> np.ndarray:
    paths: np.ndarray = np.empty([num_paths, num_steps + 1])
    dt: float = expiry / num_steps
    vol2: float = vol * vol
    for i in range(num_paths):
        paths[i, 0] = spot_price
        for step in range(num_steps):
            m: float = np.log(paths[i, step]) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            paths[i, step + 1] = np.exp(np.random.normal(m, np.sqrt(v)))
    return paths


def option_price(
    scoring_data: np.ndarray,
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    rate: float,
    strike: float
) -> float:
    num_paths: int = scoring_data.shape[0]
    num_steps: int = scoring_data.shape[1] - 1
    stoptime_rl: np.ndarray = np.zeros(num_paths)

    prices: np.ndarray = np.zeros(num_paths)
    dt: float = expiry / num_steps

    for i, path in enumerate(scoring_data):
        step: int = 0
        while step <= num_steps:
            t: float = step * dt
            exercise_price: float = max(strike - path[step], 0)
            continue_price: float = func.evaluate([(t, path[step])])[0] \
                if step < num_steps else 0.
            step += 1
            if (exercise_price >= continue_price) and (exercise_price>0):
                prices[i] = np.exp(-rate * t) * exercise_price
                stoptime_rl[i] = t 
                step = num_steps + 1

    return np.average(prices), stoptime_rl

def RL_GBM_Policy(spot_price_val,strike_val, expiry_val, rate_val
                  , vol_val, num_scoring_paths,num_steps_scoring, 
                  num_steps_lspi, num_training_paths_lspi, spot_price_frac_lspi, 
                  training_iters_lspi):
    
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

    scoring_data: np.ndarray = scoring_sim_data(
        expiry=expiry_val,
        num_steps=num_steps_scoring,
        num_paths=num_scoring_paths,
        spot_price=spot_price_val,
        rate=rate_val,
        vol=vol_val
    )

    lspi_opt_price,_ = option_price(
        scoring_data=scoring_data,
        func=flspi,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )

    return lspi_opt_price


    
#num_scoring_paths: int = 1000
#num_steps_scoring: int = 50

##num_steps_lspi: int = 10
#num_training_paths_lspi: int = 10000
#spot_price_frac_lspi: float = 0.01
#training_iters_lspi: int = 20

#spot_price_val: float = 36.0
##strike_val: float = 40.0
#expiry_val: float = 1.0
#rate_val: float = 0.06
#vol_val: float = 0.2
#num_steps_value_lsm = 50 


#Option_Price_RL=RL_GBM_Policy(spot_price_val=spot_price_val, strike_val=strike_val,
#              expiry_val=expiry_val, rate_val=rate_val,vol_val=vol_val,
#              num_training_paths_lspi=num_training_paths_lspi, 
#              spot_price_frac_lspi=spot_price_frac_lspi, 
#              training_iters_lspi=training_iters_lspi, 
#              num_scoring_paths=num_scoring_paths, 
#              num_steps_scoring=num_steps_scoring,
#              num_steps_lspi=num_steps_lspi)


#print(f"LSPI Price = {Option_Price_RL:.3f}")

#RES = []

#for ietr in np.arange(3):

#    res = RL_GBM_Policy(spot_price_val=spot_price_val, strike_val=strike_val,
##              expiry_val=expiry_val, rate_val=rate_val,vol_val=vol_val,
#              num_training_paths_lspi=num_training_paths_lspi, 
#              spot_price_frac_lspi=spot_price_frac_lspi, 
#              training_iters_lspi=training_iters_lspi, 
#              num_scoring_paths=num_scoring_paths, 
#              num_steps_scoring=num_steps_scoring,
#              num_steps_lspi=num_steps_lspi)
    
#    RES.append(res)

#RES

#####################################################################################
#####################################################################################

if __name__ == '__main__':

    S0_values_table1 = np.arange(36,46, 2)
    sd_values_table1 = np.array([0.2, 0.4])
    T_values_table1 = np.array([1, 2])



    def Table1_func(S0_values,sd_values,T_values, paths_number_train, num_scoring_paths):
        
            print("%-10s %-10s %-10s %-40s %-20s %-20s" 
                %("S0","vol", "T", "Closed Form European", "RL", 
                "Early exercise"))
        
            strike_val = 40.0
            rate_val = 0.06
            num_scoring_paths = num_scoring_paths
            num_steps_scoring = 50
            num_steps_lspi = 10
            training_iters_lspi = 20
            spot_price_frac_lspi: float = 0.001


        
            for S0_table1 in S0_values:
                for sd_table1 in sd_values:
                    for T_table1 in T_values:
                    
                        spot_price_val = S0_table1
                        expiry_val = T_table1
                        vol_val = sd_table1
                    
                        euoption = european_put_price(spot_price=spot_price_val, expiry=expiry_val,
                                                  rate=rate_val, vol=vol_val, strike=strike_val)
                    
                        Option_price = RL_GBM_Policy(spot_price_val=spot_price_val, 
                                                 strike_val=strike_val, expiry_val=expiry_val, 
                                                 rate_val=rate_val,vol_val=vol_val,
                                                 num_training_paths_lspi=paths_number_train, 
                                                 spot_price_frac_lspi=spot_price_frac_lspi, 
                                                 training_iters_lspi=training_iters_lspi, 
                                                 num_scoring_paths=num_scoring_paths, 
                                                 num_steps_scoring=num_steps_scoring,
                                                 num_steps_lspi=num_steps_lspi)

      
                        print("%d %10.2f %10d %40.3f %20.3f %20.3f" 
                            %(S0_table1,sd_table1, T_table1, euoption, 
                                Option_price,Option_price-euoption))


    Table1_func(S0_values=S0_values_table1, sd_values=sd_values_table1, 
            T_values=T_values_table1, paths_number_train=10000, 
            num_scoring_paths=1000)