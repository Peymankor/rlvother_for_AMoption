

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


def continuation_curve_lsm(left_price, right_price,deltaprice,      
                            lsm_policy_coef, K_value, expiry_val, num_steps_value_lsm):

    prices = np.arange(left_price,right_price,deltaprice)
    
    x = []
    y = []

    for time_step in reversed(lsm_policy_coef.keys()):
    
        cp = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
        lsm_policy_coef[time_step]) for p in prices])

        ep = np.array([max(K_value - p, 0) for p in prices])

        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep) if e > c]
   
        if len(ll) > 0:
            x.append((time_step+1) * expiry_val / num_steps_value_lsm)
            y.append(max(ll))

    final: Sequence[Tuple[float, float]] = \
        [(p, max(K_value - p, 0)) for p in prices]
    
    x.append(expiry_val)
    y.append(max(p for p, e in final if e > 0))

    return x, y



def continuation_curve(
    func: FunctionApprox[Tuple[float, float]],
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return func.evaluate([(t, p) for p in prices])


def exercise_curve(
    strike: float,
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return np.array([max(strike - p, 0) for p in prices])


def put_option_exercise_boundary(
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    num_steps: int,
    strike: float
) -> Tuple[Sequence[float], Sequence[float]]:
    x: List[float] = []
    y: List[float] = []
    prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y


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

if __name__ == '__main__':
    
    # Example of Longstaff og Schwartz

    spot_price_val: float = 36.0
    strike_val: float = 40.0
    expiry_val: float = 1.0
    rate_val: float = 0.06
    vol_val: float = 0.2
    num_scoring_paths: int = 10000
    num_steps_scoring: int = 50

    num_steps_lspi: int = 10
    num_training_paths_lspi: int = 10000
    spot_price_frac_lspi: float = 0.1
    training_iters_lspi: int = 8

    num_steps_value_lsm = 50 
    num_paths_train_lsm = 10000
    K_value = 40
    k_value = 4
    num_paths_test_value_lsm = 10000

    #random.seed(100)
    #np.random.seed(100)

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

    print("Fitted LSPI Model")

    european_price: float = european_put_price(
        spot_price=spot_price_val,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val
    )

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike_val - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=100
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    bin_tree_price: float = vf_seq[0][NonTerminal(0)]
    bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
    bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)

    lspi_x, lspi_y = put_option_exercise_boundary(
        func=flspi,
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        strike=strike_val
    )

    # LSM method


    def payoff_func(_: float, s: float) -> float:
            return max(K_value - s, 0.)


    lsmclass = OptimalExerciseLSM(spot_price=spot_price_val,
                 payoff=payoff_func, expiry=expiry_val,
                 rate=rate_val, vol=vol_val,
                 num_steps=num_steps_value_lsm)

    train_data_v = lsmclass.GBMprice_training(
    num_paths_train=num_paths_train_lsm, seed_random=False)

    lsm_policy_coef_beta = lsmclass.train_LSM(training_data=train_data_v,
                            num_paths_train=num_paths_train_lsm, 
                            K=K_value, k=k_value)
    
    print("Fitted LSM Model")
 
    test_data_v = lsmclass.scoring_sim_data(num_paths_test=
                    num_paths_test_value_lsm)

    
    lsm_opt_price, _ = lsmclass.option_price(scoring_data=test_data_v, 
                        Beta_list=lsm_policy_coef_beta,
                            k=k_value)
    

    num_paths_test_value_lsm_stoptime = 10

    test_data_v_stoptime = lsmclass.scoring_sim_data(num_paths_test=
                        num_paths_test_value_lsm_stoptime)

    _, lsm_stoptimes = lsmclass.option_price(scoring_data=test_data_v_stoptime, 
                        Beta_list=lsm_policy_coef_beta,
                            k=k_value)
    
    
    dt = expiry_val/num_steps_value_lsm
    taxis = np.arange(dt,1+dt,dt)
    

    #fig, ax = plt.subplots(figsize=(11, 7))
    #for price_path in np.arange(len(test_data_v_stoptime)):
        
    #    price_list = test_data_v_stoptime[price_path]
        
    #    color = next(ax._get_lines.prop_cycler)['color']
    #    ax.plot(taxis, price_list, color = color)        
        
    #    stopindex = (lsm_stoptimes[price_path]*num_steps_value_lsm -1)
    #    ax.plot(lsm_stoptimes[price_path], price_list[int(stopindex)], marker = "*", color=color, markersize=10)

    #ax.set_xlabel ("Time (Normalized)", fontsize=20)
    #ax.set_ylabel("Underlying Price", fontsize=20)
    #ax.grid(True)
    #ax.set_title("Stopping Time at Test Paths", fontsize=25)
    #plt.show()

    lsm_bound_x, lsm_bound_y = continuation_curve_lsm(
                    left_price=20,right_price=40, 
                    deltaprice=0.1, 
                    lsm_policy_coef=lsm_policy_coef_beta, 
                    K_value=K_value, expiry_val=expiry_val, 
                    num_steps_value_lsm= num_steps_value_lsm
    )

    print("Plotting boundary curve in Bin Tree method, Ex1 LSM paper")

    plot_list_of_curves(
        list_of_x_vals=[bin_tree_x],
        list_of_y_vals=[bin_tree_y],
        list_of_colors=["g"],
        list_of_curve_labels=["Binary Tree"],
        x_label="Time",
        y_label="Underlying Price",
        title="Binary Tree Exercise Boundaries"
    )


    print("Boundary curve in Bin Tree method and LSPI, Ex1 LSM paper")


    plot_list_of_curves(
        list_of_x_vals=[lspi_x, bin_tree_x],
        list_of_y_vals=[lspi_y, bin_tree_y],
        list_of_colors=["b", "g"],
        list_of_curve_labels=["LSPI", "Binary Tree"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, Binary Tree Exercise Boundaries"
    )


    print("Boundary curve in Bin Tree method and LSPI, LSM Ex1 LSM paper")

    plot_list_of_curves(
        list_of_x_vals=[lspi_x, bin_tree_x, lsm_bound_x],
        list_of_y_vals=[lspi_y, bin_tree_y, lsm_bound_y],
        list_of_colors=["b", "g", "r"],
        list_of_curve_labels=["LSPI", "Binary Tree", "LSM"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, Binary Tree and LSM Exercise Boundaries"
    )


    scoring_data: np.ndarray = scoring_sim_data(
        expiry=expiry_val,
        num_steps=num_steps_scoring,
        num_paths=num_scoring_paths,
        spot_price=spot_price_val,
        rate=rate_val,
        vol=vol_val
    )

    scoring_data_stoptime = np.c_[np.ones(num_paths_test_value_lsm_stoptime)*spot_price_val,test_data_v_stoptime]
    
    _, rl_stoptimes = option_price(
        scoring_data=scoring_data_stoptime,
        func=flspi,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )


    fig, ax = plt.subplots(figsize=(11, 7))
    for price_path in np.arange(len(test_data_v_stoptime)):
        
        price_list = test_data_v_stoptime[price_path]
        
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(taxis, price_list, color = color)        
        

        stopindex = (lsm_stoptimes[price_path]*num_steps_value_lsm -1)
        ax.plot(lsm_stoptimes[price_path], price_list[int(stopindex)], marker = "x", color=color, 
                markersize=15)

        stopindex = (rl_stoptimes[price_path]*num_steps_value_lsm -1)
        ax.plot(rl_stoptimes[price_path], price_list[int(stopindex)], marker = "o",     
                color=color, markersize=15, fillstyle= "none", label="Test11")

    ax.set_xlabel ("Time (Normalized)", fontsize=20)
    ax.set_ylabel("Underlying Price", fontsize=20)
    ax.grid(True)
    ax.set_title("Stopping Time at Test Paths, Cross (LSM), Circle (RL)", fontsize=25)
    plt.show()

    #plt.show()
    lspi_opt_price,_ = option_price(
        scoring_data=scoring_data,
        func=flspi,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )


    #print(f"LSPI Option Price = {lspi_opt_price:.3f}")
    
    print(f"European Put Price = {european_price:.3f}")
    print(f"Binary Tree Price = {bin_tree_price:.3f}")
    print(f"LSM Price = {lsm_opt_price:.3f}")
    print(f"LSPI Price = {lspi_opt_price:.3f}")

##################################################