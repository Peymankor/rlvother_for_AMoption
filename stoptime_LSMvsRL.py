

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

num_scoring_paths_val: int = 10000
num_steps_scoring: int = 50
num_steps_lspi: int = 10
num_training_paths_lspi_val: int = 1000
spot_price_frac_lspi: float = 0.000000001
training_iters_lspi_val: int = 10

S0_values_table1 = np.array([36])
sd_values_table1 = np.array([0.2])
T_values_table1 = np.array([2])

spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 2.0
rate_val: float = 0.06
vol_val: float = 0.2

num_steps_value_lsm = 50 

K_value = 40
k_value = 4
num_paths_test_val = 10000
num_paths_train_val = 100000


import matplotlib.pyplot as plt


price_lsm, stop_values_lsm = lsm_price(spot_price_val=S0_values_table1, strike_val=strike_val,
                                expiry_val=T_values_table1, rate_val=rate_val, 
                                vol_val=vol_val, steps_value=num_steps_value_lsm,
                                    K_value=K_value, k_value=k_value, 
                                    num_paths_train_val=num_paths_train_val,
                                    num_paths_test_val=num_paths_test_val)


#########################################################################


price_rl, stop_values_rl = RL_GBM_Policy(spot_price_val=spot_price_val, strike_val=strike_val, 
                         expiry_val=expiry_val,
                      rate_val=rate_val, vol_val=vol_val, num_scoring_paths=num_scoring_paths_val,
                      num_steps_scoring=num_steps_scoring, num_steps_lspi=num_steps_lspi,
                      num_training_paths_lspi=num_training_paths_lspi_val,spot_price_frac_lspi=spot_price_frac_lspi,
                      training_iters_lspi=training_iters_lspi_val)



##############################################################################

chart1_dataset1 = pd.DataFrame({"RL":stop_values_rl,
                                "LSM":stop_values_lsm})


chart_1_dataset_melted1 = pd.melt(chart1_dataset1, value_vars=["RL", "LSM"],
                                                            var_name="Method",
                                                            value_name="stoptime")


#from plotnine import ggplot, labs, aes, scale_x_continuous, theme, geom_bar, position_dodge,geom_histogram
from plotnine import *

xx=(ggplot(chart_1_dataset_melted1, aes(x = "stoptime", fill = "Method")) + 
  geom_histogram(position="dodge2", binwidth = 0.5 , center = 0) +
  scale_x_continuous(breaks=range(0, 51, 1)) +
  labs(y='Frequency', x="The Stopping Time Step (k)") +
  #scale_fill_manual(legend_title,values=c("orange","red")) +
theme(legend_position= (0.87, 0.75),
      text = element_text(size = 14)))

ggsave(self=xx, filename='plot_T_2.png', width=16,
       height=8)

#geom_bar(position = "dodge2"))
# geom_histogram(stat="count", position="dodge", alpha = 0.8))

