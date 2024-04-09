################# Plot the Frequency Plot ########################

## Initial import ############

from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
import pandas as pd

from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_bar, position_dodge



import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sys 
from rich import print, pretty
pretty.install()

## EU option

sys.path.append("..")

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
num_steps_lspi: int = 15
num_training_paths_lspi_val: int = 10000
spot_price_frac_lspi: float = 0.0000000000001
training_iters_lspi_val: int = 8

S0_values_table1 = np.array([36])
sd_values_table1 = np.array([0.4])
T_values_table1 = np.array([1])

spot_price_val: float = 36.0
strike_val: float = 40.0
expiry_val: float = 1.0
rate_val: float = 0.06
vol_val: float = 0.4

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
stop_values_lsm

price_rl, stop_values_rl = RL_GBM_Policy(spot_price_val=spot_price_val, strike_val=strike_val, 
                         expiry_val=expiry_val,
                      rate_val=rate_val, vol_val=vol_val, num_scoring_paths=num_scoring_paths_val,
                      num_steps_scoring=num_steps_scoring, num_steps_lspi=num_steps_lspi,
                      num_training_paths_lspi=num_training_paths_lspi_val,spot_price_frac_lspi=spot_price_frac_lspi,
                      training_iters_lspi=training_iters_lspi_val)

stop_values_rl
#stop_values_rl_mixed = stop_values_rl.astype(object)
#stop_values_rl_mixed[stop_values_rl==100] = "Not exercised"
#stop_values_rl_mixed

#stop_values_lsm_mixed = stop_values_lsm.astype(object)
#stop_values_lsm_mixed[stop_values_lsm==0] = "Not exercised"

#stop_values_lsm_mixed

##############################################################################

chart1_dataset1 = pd.DataFrame({"Q-Learning (RL)":stop_values_rl,
                                "LSM":stop_values_lsm})


chart_1_dataset_melted1 = pd.melt(chart1_dataset1, value_vars=["Q-Learning (RL)", "LSM"],
                                                            var_name="Method",
                                                            value_name="stoptime")

chart_1_dataset_melted1
#from plotnine import ggplot, labs, aes, scale_x_continuous, theme, geom_bar, position_dodge,geom_histogram
from plotnine import *

xx=(ggplot(chart_1_dataset_melted1, aes(x = "stoptime", fill = "Method")) + 
  geom_histogram(position="dodge2", binwidth = 0.5 , center = 0) +
  scale_x_continuous(breaks=range(0, 51, 1)) +
  labs(y='Frequency (from total of 10,000 Paths)', x="The Stopping Time Step (k)") +
  #scale_fill_manual(legend_title,values=c("orange","red")) +
theme(legend_position= (0.8, 0.75),
      axis_text_x=element_text(angle=90),
      text = element_text(size = 7)) +
      ggtitle("S0 = 36, Strike Price = 40, Volatility = 0.4"))
#xx.save('plot_T_21.png', dpi = 600)
ggsave(self=xx, filename='Results/Fig/Freq_S036StrikePrice40Volatility0.4.png', width= 6, height = 4, dpi = 600)
price_rl
xx
#geom_bar(position = "dodge2"))
# geom_histogram(stat="count", position="dodge", alpha = 0.8))

