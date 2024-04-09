import sys 
from rich import print, pretty
pretty.install()

## EU option
import os
os.getcwd()
import numpy as np

sys.path.append("..")


from final_policies_GARCH_back import OptionPricingRL_GARCH


num_scoring_paths_f = 10000
num_train_paths_val = 5000
    
Test = OptionPricingRL_GARCH(spot_price=80, strike=80, expiry=1, 
                              num_time_steps=50, num_train_paths=num_train_paths_val, 
                              rate=0.06)
Train_data = Test.generate_garch_rl_training_data()
func_rl = Test.fitted_lspi_put_option_GARCH(Train_data, training_iters=3)
func_rl.evaluate([(0, 80)])

scoring_data = Test.generate_garch_rl_scoring_data(num_scoring_paths=num_scoring_paths_f)

final_day_price = Test.calculate_final_day_price(num_scoring_paths=num_scoring_paths_f)
option_price_rl, stoptime_rl = Test.option_price(scoring_data, func_rl)

print("Option price by RL in GARCH: ", option_price_rl)
print("Final Day Price in GARCH: ", final_day_price)