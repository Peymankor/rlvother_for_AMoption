import sys 
from rich import print, pretty
pretty.install()

## EU option

import numpy as np

sys.path.append("..")

from RL_policy import training_sim_data, fitted_lspi_put_option, scoring_sim_data , option_price

from typing import Callable, Dict, Sequence, Tuple, List

from helper import continuation_curve_lsm, put_option_exercise_boundary
import numpy as np
import pandas as pd


expiry_val = 1
num_steps_val = 10
num_paths_val = 5000
spot_price_val = 36
spot_price_frac_val = 0.0000000000000000000000000001
rate_val = 0.06
vol_val = 0.2
strike_val = 40
training_iters_val = 4
num_steps_val_score = 50
dt_val = expiry_val / num_steps_val_score
price_point_val = 30

fited_lspi = fitted_lspi_put_option(expiry=expiry_val,
                                    num_steps=num_steps_val,num_paths=num_paths_val,
                                    spot_price=spot_price_val,
                                    spot_price_frac=spot_price_frac_val,
                                    rate=rate_val,vol=vol_val,
                                    strike=strike_val,
                                    training_iters=training_iters_val)

#x_10 = training_sim_data(expiry=expiry_val, num_steps=num_steps_val, num_paths=num_paths_val,
#                  spot_price=spot_price_val, rate=rate_val, vol=vol_val, 
#                  spot_price_frac=spot_price_frac_val)

#print("With 10 time steps")
#print(x_10)
#x_10_array = np.array(x_10)
#filtered_x_10 = x_10_array[x_10_array[:, 0] == 9]
#filtered_x_10
#endprices =filtered_x_10[:, 2]

#import matplotlib.pyplot as plt

# ... existing code ...

# Generate histogram of end prices
##plt.hist(endprices, bins=20)
##plt.xlabel('End Prices')
#plt.ylabel('Frequency')
#plt.title('Histogram of End Prices')
#plt.show()


#print(f"Standard Deviation of x_10 data when the tuple starts with 9: {std_dev}")


##size_x_10 = len(x_10)
#print(f"Size of x_50 list: {size_x_10}")

#x_50 = training_sim_data(expiry=expiry_val, num_steps=50, num_paths=num_paths_val,
#                  spot_price=spot_price_val, rate=rate_val, vol=vol_val, 
#                  spot_price_frac=spot_price_frac_val)
#size_x_50 = len(x_50)
#print(f"Size of x_50 list: {size_x_50}")

#print("With 50 time steps")
#print(x_50)

def generate_next_prices(N_sample: int, price: float, vol: float, 
                         rate: float, dt: float) -> List[float]:
    vol2 = vol * vol
    m = np.log(price) + (rate - vol2 / 2) * dt
    v = vol2 * dt
    next_prices = []
    
    while len(next_prices) < N_sample:
        next_price = np.exp(np.random.normal(m, np.sqrt(v)))
        next_prices.append(next_price)
        price = next_price
    
    return next_prices

# N_sample_val = 100000

# next_prices_val = generate_next_prices(N_sample=N_sample_val, 
#                                        price=price_point_val, vol=vol_val, 
#                                        rate=rate_val, 
#                                        dt= dt_val)

# result_eu = np.maximum(strike_val - np.array(next_prices_val), 0)
# RL_step_before = fited_lspi.evaluate([(0.98, price_point_val)])[0]

# print(f"RL_step_before: {RL_step_before}")
# print(f"result_eu: {result_eu.mean()}")


# option_prices = []

# for _ in range(20):
#     test_data = scoring_sim_data(expiry=expiry_val, num_steps=num_steps_val_score, num_paths=5000,
#                                  spot_price=spot_price_val, rate=rate_val, vol=vol_val)

#     price_RL, stop_time = option_price(test_data, fited_lspi, expiry=expiry_val,
#                                        rate=rate_val,
#                                        strike=strike_val)

#     option_prices.append(price_RL)

# #print(test_data)
# #test_data.shape

# average_option_price = sum(option_prices) / len(option_prices)
# print(f"Average Option Price: {average_option_price}")

# #print(f"Option Price: {price_RL}")

# test_data = scoring_sim_data(expiry=expiry_val, num_steps=num_steps_val_score, num_paths=5000,
#                                  spot_price=spot_price_val, rate=rate_val, vol=vol_val)

# test_data[:,-1]

# # Generate histogram of end prices
# import matplotlib.pyplot as plt
# plt.hist(test_data[:,-1], bins=20)
# plt.xlabel('End Prices')
# plt.ylabel('Frequency')
# plt.title('Histogram of End Prices')
# plt.show()

# price_RL, stop_time = option_price(test_data, fited_lspi, expiry=expiry_val,
#                                        rate=rate_val,
#                                        strike=strike_val)


# #price_RL
# #stop_time
# import matplotlib.pyplot as plt

# plt.hist(stop_time, bins=range(1, 53))  # Increase the range to include 51
# plt.xlabel('Price')
# plt.ylabel('Frequency')
# plt.title('Histogram of Price_RL')

# # Add x-axis labels for each bin
# plt.xticks(range(1, 52))

# plt.show()

# lspi_x, lspi_y = put_option_exercise_boundary(
#         func=fited_lspi,
#         expiry=1,
#         num_steps=50,
#         strike=40
#     )

# # plot lspi_x, lspi_y
# import matplotlib.pyplot as plt
# plt.scatter(lspi_x, lspi_y)
# plt.show()

####################################################################
####################################################################
from LSM_policy import OptimalExerciseLSM
from basis_fun import laguerre_polynomials, laguerre_polynomials_ind
import pandas as pd
import pandas as pd

num_steps_val = 50

def payoff_func(_: float, s: float) -> float:
            return max(strike_val - s, 0.)

Testclass = OptimalExerciseLSM(spot_price=spot_price_val, payoff=payoff_func,expiry=expiry_val,
                                        rate=rate_val, vol=vol_val,
                                        num_steps=num_steps_val)

train_data_v = Testclass.GBMprice_training(num_paths_train=num_paths_val)

k_value = 4
lsm_policy_v = Testclass.train_LSM(training_data=train_data_v, num_paths_train=num_paths_val,
                                            K=strike_val, k=k_value)

step_point = 49

#XX=laguerre_polynomials_ind(np.array([price_point_val]),k_value)
#result_matrix = np.dot(XX.T, lsm_policy_v[step_point-1])


#print(f"LSM one step before: {result_matrix}")


##########################################################################
##########################################################################

df = pd.DataFrame(columns=['Price', 'RL_Step_Before', 'Result_EU', 'Result_Matrix'])

N_sample_val = 100000


price_list = []
rl_step_before_list = []
result_eu_list = []
LSM_list = []

for price_point_val in range(20, 40):
    next_prices_val = generate_next_prices(N_sample=N_sample_val, 
                                           price=price_point_val, vol=vol_val, 
                                           rate=rate_val, 
                                           dt=dt_val)

    result_eu = np.maximum(strike_val - np.array(next_prices_val), 0)
    RL_step_before = fited_lspi.evaluate([(0.98, price_point_val)])[0]

    #print(f"Price: {price_point_val}")
    #print(f"RL_step_before: {RL_step_before}")
    #print(f"result_eu: {result_eu.mean()}")

    step_point = 49

    XX = laguerre_polynomials_ind(np.array([price_point_val]), k_value)
    result_matrix = np.dot(XX.T, lsm_policy_v[step_point - 1])

    #print(f"LSM one step before: {result_matrix}")
    
    # Append the results to the lists
    price_list.append(price_point_val)
    rl_step_before_list.append(RL_step_before)
    result_eu_list.append(result_eu.mean())
    LSM_list.append(result_matrix.item())


df = pd.DataFrame({'Price': price_list, 
                   'RL_Step_Before': rl_step_before_list, 
                   'Result_EU': result_eu_list,
                   "Result_LSM": LSM_list})
print(df)

#pd.__version__

import matplotlib.pyplot as plt

# Extract the data from the DataFrame
price = df['Price']
rl_step_before = df['RL_Step_Before']
result_eu = df['Result_EU']
result_lsm = df['Result_LSM']

# Set the width of the bars
bar_width = 0.25

# Set the positions of the bars on the x-axis
r1 = np.arange(len(price))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot the bars
plt.bar(r1, rl_step_before, color='blue', width=bar_width, edgecolor='black', label='RL')
plt.bar(r2, result_eu, color='green', width=bar_width, edgecolor='black', label='EU')
plt.bar(r3, result_lsm, color='orange', width=bar_width, edgecolor='black', label='LSM')

# Add labels and title
plt.xlabel('Price')
plt.ylabel('Value')
plt.title('Comparison of Continuation value based on RL and LSM method and \
          EU Value option, one step before expiry time')

# Add x-axis ticks and labels
plt.xticks([r + bar_width for r in range(len(price))], price)

# Add a legend
plt.legend()

# Show the plot
plt.show()


