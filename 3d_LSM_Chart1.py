

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


