from RL_policy_GARCH import RL_GBM_Policy_GARCH
from price_model import Brent_GARCH_light
from helper import data_to_rlform, EU_payoff_from_last_col
import numpy as np

expiry_val_f = 1
rate_val_f = 0.06
K_value = strike_val= 90
k_value = 4

option_horizon_indays_f = 365
initial_price_val_f = 80
num_time_steps_val_f = 10
steps_value_f = num_time_steps_val_f-1
num_path_numbers_train_f = 10000

# garch price

garch_price_class = Brent_GARCH_light()

dataset_train = garch_price_class.garch_price_creator(
     option_horizon_indays=option_horizon_indays_f,
     num_time_steps_val=num_time_steps_val_f,
     num_path_numbers_val=num_path_numbers_train_f,
     initial_price_val=initial_price_val_f)



rl_garch_train = data_to_rlform(dataset=dataset_train, 
               initial_price_val_f=initial_price_val_f)

#print(rl_garch_train)
##############

num_path_numbers_test_f = 10000
rl_garch_test = garch_price_class.garch_price_creator(
     option_horizon_indays=option_horizon_indays_f,
     num_time_steps_val=num_time_steps_val_f,
     num_path_numbers_val=num_path_numbers_test_f,
     initial_price_val=initial_price_val_f)

###############

#print(rl_garch_test)


from RL_policy_GARCH import RL_GBM_Policy_GARCH

num_steps_lspi_f = 10
training_iters_lspi_val_f = 10



price_rl_garch = RL_GBM_Policy_GARCH(expiry_val=expiry_val_f,
                                     num_steps_lspi=num_steps_lspi_f,
                                     rate_val=rate_val_f,
                                     strike_val=strike_val,
                                     training_iters_lspi= training_iters_lspi_val_f,
                                     training_data_val=rl_garch_train,
                                     scoring_data_val=rl_garch_test)

price_eu_garch = EU_payoff_from_last_col(strike_value=strike_val,
                                         last_column=rl_garch_test[:,-1])

print('{} = {}'.format("Option Price Using RL is ", price_rl_garch))
print('{} = {}'.format("Option Price Using EU is ", price_eu_garch))

#print(10)

#formatted