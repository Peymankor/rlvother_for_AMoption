import numpy as np
from basis_fun import laguerre_polynomials_ind
from typing import Sequence
from typing import Callable, Dict, Sequence, Tuple, List
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox

def data_to_rlform(dataset, initial_price_val_f):
    
    rett = []

    for _ ,path in enumerate(dataset):

        price = initial_price_val_f

        for step, price_ele in enumerate(path):

            rett.append((step, price, price_ele))
        
            price = price_ele
    
    return(rett)


def EU_payoff_from_last_col(strike_value, last_column):
        
        payoff_end = strike_value-last_column
        payoff_end_pos = (strike_value-last_column>0)

        option_value = np.sum(payoff_end[payoff_end_pos]*np.exp(-0.06*1))/len(last_column)


        return option_value


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

    prices: np.ndarray = np.arange(20, strike + 0.1, 0.1)
    
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        #print(cp)
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            #print(x)
            y.append(max(ll))
            #print(y)
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y

def garch_array_to_list(garhc_prices_array):
    
    garch_prices_list = []
    
    _, Nsteps = garhc_prices_array.shape
    
    for row in garhc_prices_array:

        for i in range(Nsteps-1):
            garch_prices_list.append((i+ 1, row[i], row[i+1]))
    
    return garch_prices_list