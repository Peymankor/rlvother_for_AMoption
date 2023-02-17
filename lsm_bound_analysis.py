import numpy as np
from LSM_policy import OptimalExerciseLSM
from basis_fun import laguerre_polynomials_ind
from typing import Callable, Dict, Sequence, Tuple
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

spot_price_value = 36
expiry_value = 1
rate_value = 0.06
vol_value = 0.2
num_steps_value = 50 
num_paths_train_value = 10000
K_value = 40
k_value = 4
num_paths_test_value = 1000


def payoff_func(_: float, s: float) -> float:
            return max(K_value - s, 0.)

lsmclass = OptimalExerciseLSM(spot_price=spot_price_value,
                                payoff=payoff_func,
                                expiry=expiry_value,
                                rate=rate_value, vol=vol_value,
                                num_steps=num_steps_value)

train_data_v = lsmclass.GBMprice_training(
    num_paths_train=num_paths_train_value)

lsm_policy_coef = lsmclass.train_LSM(training_data=train_data_v,
                            num_paths_train=num_paths_train_value, K=K_value,
                            k=k_value)

test_data_v = lsmclass.scoring_sim_data(num_paths_test=
                    num_paths_test_value)


Option_price = lsmclass.option_price(scoring_data=test_data_v, 
                        Beta_list=lsm_policy_coef,k=k_value)

print(f"LSM Put Price = {Option_price:.3f}")

prices = np.arange(20,40,1)
x=[]
y = []

for time_step in reversed(lsm_policy_coef.keys()):
    
    cp = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
    lsm_policy_coef[time_step]) for p in prices])

    ep = np.array([max(K_value - p, 0) for p in prices])

    ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep) if e > c]
   
    if len(ll) > 0:
        x.append((time_step+1) * expiry_value / num_steps_value)
        y.append(max(ll))

final: Sequence[Tuple[float, float]] = \
[(p, max(K_value - p, 0)) for p in prices]
x.append(expiry_value)
y.append(max(p for p, e in final if e > 0))

plt.plot(x, y)
plt.show()


