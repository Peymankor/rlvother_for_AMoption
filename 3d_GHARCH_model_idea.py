################ 
# This code try to calculate option value for the case underlying price has a 
# GARCH model


# First make GARCh(1,1) price model
import pandas as pd
from price_model import Brent_GARCH_light
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from LSM_policy import OptimalExerciseLSM
from LSM_policy import  lsm_price
import numpy as np


#################################################
#################################################

# How to import dataframe
# Brent_crude_df = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", 
#                        sheet_name="Data 1")

# how to save your dataframe

# Brent_crude_df.to_csv("data/raw_data/Brent_til_31July.csv",
#                      index = False)

#################################################
#################################################

# Initial data

spot_price_val_f = "XX"
expiry_val_f = 1
rate_val_f = 0.06
vol_val_f = "WW"
steps_value_f = 50-1
K_value = strike_val= 90
k_value = 4

option_horizon_indays_f = 365
initial_price_val_f = 80
num_time_steps_val_f = 50
steps_value_f = num_time_steps_val_f-1
num_path_numbers_train_f = 1000
num_path_numbers_test_f = 100

# garch price

garch_price_class = Brent_GARCH_light()

dataset_train = garch_price_class.garch_price_creator(
     option_horizon_indays=option_horizon_indays_f,
     num_time_steps_val=num_time_steps_val_f,
     num_path_numbers_val=num_path_numbers_train_f,
     initial_price_val=initial_price_val_f)

dataset_train


def payoff_func(_: float, s: float) -> float:
     return max(strike_val - s, 0.)

garchclass = OptimalExerciseLSM(spot_price=spot_price_val_f, payoff=payoff_func,
                               expiry=expiry_val_f,rate=rate_val_f, vol=vol_val_f,
                               num_steps=steps_value_f)



lsm_policy_v_garch = garchclass.train_LSM(training_data=dataset_train, 
                                   num_paths_train=num_path_numbers_train_f,
                                   K=K_value, k=k_value)
lsm_policy_v_garch

dataset_test = garch_price_class.garch_price_creator(
     option_horizon_indays=option_horizon_indays_f,
     num_time_steps_val=num_time_steps_val_f,
     num_path_numbers_val=num_path_numbers_test_f,
     initial_price_val=initial_price_val_f)


Option_price_garch,_ = garchclass.option_price(scoring_data=dataset_test,
                                               Beta_list=lsm_policy_v_garch, k=k_value)
Option_price_garch


Option_price_garch
max(dataset_test[:,-1]-strike_val,0)
np.shape(dataset_test)
payoff_end = strike_val-dataset_test[:,-1]
payoff_end
payoff_end_pos = (strike_val-dataset_test[:,-1]>0)
payoff_end_pos

np.sum(payoff_end[payoff_end_pos]*np.exp(-0.06*1))/100



payoff_end[payoff_end]
Option_price_garch
#####################################################################
#####################################################################
import matplotlib.pyplot as plt
plt.plot(dataset_test.T)
plt.show()
garch_prices

100*np.exp(-0.06*2)

100/(0.06 +1)**2

#garch_prices

#len(garch_prices)
#garch_prices



TEST = Brent_GARCH(brent_df=Brent_crude_df)

TEST.compare_model()



## Initial import ############

from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
import pandas as pd

from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_bar, position_dodge



import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from rich import print, pretty
pretty.install()

## EU option

from eu_opt import european_put_price

## Import packages for Bin

from bintree_policy import bin_tree_price

## Import packages for LSM

from LSM_policy import  lsm_price

## Import packages for RL

from RL_policy import RL_GBM_Policy


################################################################################################



if __name__ == "__main__":

    spot_price_val: float = 36.0
    strike_val: float = 40.0
    expiry_val: float = 1.0
    rate_val: float = 0.06
    vol_val: float = 0.2



    num_steps_value_lsm = 50 
    num_paths_train_val = 100000
    K_value = 40
    k_value = 4
    num_paths_test_val = 10000


    num_scoring_paths_val: int = 5000
    num_steps_scoring: int = 50
    num_steps_lspi: int = 10
    num_training_paths_lspi_val: int = 1000
    spot_price_frac_lspi: float = 0.000000001
    training_iters_lspi_val: int = 10

    S0_values_table1 = np.array([36,38,40,42, 44])
    sd_values_table1 = np.array([0.2, 0.4])
    T_values_table1 = np.array([1,2])

    eu_price_list =[]
    bin_price_list = []
    lsm_price_list = []
    rl_price_list = []
    S0_list = []
    vol_list = []
    T_list = []

    print("%-10s %-10s %-10s %-20s %-20s %-20s %-20s" 
                          %("S0","vol", "T", "EU", "Bin Tree", "LSM", "RL"))
    for S0_table1 in S0_values_table1:
            for sd_table1 in sd_values_table1:
                for T_table1 in T_values_table1:
                     

                     eu_price = european_put_price(spot_price=S0_table1, expiry=T_table1, vol=sd_table1,
                                                   rate=rate_val, strike=strike_val)
                     
                     
                     price_bin = bin_tree_price(spot_price_val=S0_table1, strike_val=strike_val,
                                         expiry_val=T_table1, rate_val=rate_val, vol_val=sd_table1)
                     

                     price_lsm = lsm_price(spot_price_val=S0_table1, strike_val=strike_val,
                                expiry_val=T_table1, rate_val=rate_val, vol_val=sd_table1, 
                                    steps_value=num_steps_value_lsm,
                                    K_value=K_value, k_value=k_value, num_paths_train_val=num_paths_train_val,
                                    num_paths_test_val=num_paths_test_val)
                     

                     price_rl = RL_GBM_Policy(spot_price_val=S0_table1, strike_val=strike_val, expiry_val=T_table1,
                      rate_val=rate_val, vol_val=sd_table1, num_scoring_paths=num_scoring_paths_val,
                      num_steps_scoring=num_steps_scoring, num_steps_lspi=num_steps_lspi,
                      num_training_paths_lspi=num_training_paths_lspi_val,spot_price_frac_lspi=spot_price_frac_lspi,
                      training_iters_lspi=training_iters_lspi_val)
                     


                     eu_price_list.append(eu_price)
                     bin_price_list.append(price_bin)
                     lsm_price_list.append(price_lsm)
                     rl_price_list.append(price_rl)
                     
                     S0_list.append(S0_table1)
                     vol_list.append(sd_table1)
                     T_list. append(T_table1)

                     print("%d %10.2f %10d %20.3f %20.3f %20.3f %20.3f" 
                        %(S0_table1,sd_table1, T_table1, eu_price, 
                            price_bin,price_lsm, price_rl))


eu_price_list_arr = np.array(eu_price_list)
bin_price_list_arr = np.array(bin_price_list)
lsm_price_list_arr = np.array(lsm_price_list)
rl_price_list_arr = np.array(rl_price_list)

vol_list_arr = np.array(vol_list)
S0_list_arr = np.array(S0_list)
T_list_arr = np.array(T_list)


plt.hist(lsm_price_list_arr- bin_price_list_arr, alpha = 0.5, label ="LSM minus BIN")
plt.hist(rl_price_list_arr- bin_price_list_arr, alpha = 0.5, label ="RL minus BIN")

plt.legend()
plt.show()
print(np.sum((rl_price_list_arr-lsm_price_list_arr)**2))

#np.sum(abs(rl_price_list_arr-lsm_price_list_arr)**2)
chart1_dataset = pd.DataFrame({"S0":S0_list_arr,  "vol":vol_list_arr, "T":T_list_arr,  
                                "D-EU_value": eu_price_list_arr, "A-Bin_value":bin_price_list_arr,
                                "B-LSM_val": lsm_price_list_arr, "C-RL_val":rl_price_list_arr})
chart1_dataset

import dataframe_image as dfi

dfi.export(chart1_dataset, 'Example1_LSM_Paper.png')

chart_1_dataset_melted = pd.melt(chart1_dataset, id_vars=["vol", "S0"], value_vars=["D-EU_value", "A-Bin_value",
                                                            "B-LSM_val", "C-RL_val"],
                                                            var_name="Method",
                                                            value_name="Option Value")

rounded_df = chart1_dataset.round(3)

#rounded_numbers = [
#round(num, 3)  for num  in numbers]

markdown_tb = rounded_df.to_markdown(index=False)
print(markdown_tb)

plot_of_res =(ggplot(chart_1_dataset_melted, aes("S0", "Option Value", color="factor(Method)"))
 +  geom_bar(stat="identity", position=position_dodge())
 + facet_wrap("~vol"))

print(plot_of_res)


