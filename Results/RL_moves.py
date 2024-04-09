
######################
# Path: Results/RL_moves.py


import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_histogram, scale_x_continuous, facet_grid, labs, \
theme, element_text, ggtitle, scale_y_continuous, geom_bar, ggsave, element_text
from typing import Callable, Tuple, List
from numpy.polynomial.laguerre import lagval
from rl.function_approx import LinearFunctionApprox, Weights
from RL_policy import training_sim_data, option_price, scoring_sim_data
from rich import print, pretty

TrainingDataType = Tuple[int, float, float]


pretty.install()


class OptionPricing_RL_move:
    def __init__(self, num_scoring_paths_val: int,
                 num_training_paths_lspi_val: int,
                 num_steps_lspi_val: int,
                 num_training_iters_lspi_val : int,
                 vol_val: int):

        self.num_scoring_paths_val = num_scoring_paths_val
        self.num_training_paths_lspi_val = num_training_paths_lspi_val
        self.num_steps_scoring = 50
        self.num_steps_lspi = num_steps_lspi_val
        self.spot_price_frac_lspi = 0.000000000000000001
        self.training_iters_lspi_val = num_training_iters_lspi_val

        self.spot_price_val = 36.0
        self.strike_val = 40.0
        self.expiry_val = 1.0
        self.rate_val = 0.06
        self.vol_val = vol_val

        self.k_value = 4

    def fitted_lspi_put_option_iteration(
        self,
        expiry: float,
        num_steps: int,
        num_paths: int,
        spot_price: float,
        spot_price_frac: float,
        rate: float,
        vol: float,
        strike: float,
        training_iters: int,
    ) -> LinearFunctionApprox[Tuple[float, float]]:
        
        num_laguerre: int = 5
        epsilon: float = 1e-3
    #print(epsilon)

        ident: np.ndarray = np.eye(num_laguerre)
        features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
        features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    
        features += [
            (lambda t_s, i=i: np.exp(-t_s[0] / (2 * expiry)) * lagval(t_s[0] / expiry, ident[i]))
            for i in range(num_laguerre)
            ]
        
        training_data: List[Tuple[int, float, float]] = training_sim_data(
            expiry=expiry,
            num_steps=num_steps,
            num_paths=num_paths,
            spot_price=spot_price,
            spot_price_frac=spot_price_frac,
            rate=rate,
            vol=vol,
        )

        dt: float = expiry / num_steps
        gamma: float = np.exp(-rate * dt)
        num_features: int = len(features)
        states: List[Tuple[float, float]] = [(i * dt, s) for i, s, _ in training_data]
        next_states: List[Tuple[float, float]] = [
            ((i + 1) * dt, s1) for i, _, s1 in training_data
        ]
        feature_vals: np.ndarray = np.array([[f(x) for f in features] for x in states])
        next_feature_vals: np.ndarray = np.array(
            [[f(x) for f in features] for x in next_states]
        )
        non_terminal: np.ndarray = np.array(
            [i < num_steps - 1 for i, _, _ in training_data]
        )
        exer: np.ndarray = np.array([max(strike - s1, 0) for _, s1 in next_states])
        #wts: np.ndarray = np.zeros(num_features)
        wts = np.ones(num_features)
        #wts = np.random.rand(num_features)

        result_history = {}

        for iteration_number in range(training_iters):
            a_inv: np.ndarray = np.eye(num_features) / epsilon
            b_vec: np.ndarray = np.zeros(num_features)
            cont: np.ndarray = np.dot(next_feature_vals, wts)
            cont_cond: np.ndarray = non_terminal * (cont > exer)
            for i in range(len(training_data)):
                phi1: np.ndarray = feature_vals[i]
                phi2: np.ndarray = phi1 - cont_cond[i] * gamma * next_feature_vals[i]
                temp: np.ndarray = a_inv.T.dot(phi2)
                a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
                b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
            wts = a_inv.dot(b_vec)
            print(wts)
            result_history[iteration_number] = LinearFunctionApprox.create(
                feature_functions=features, weights=Weights.create(wts)
            )

        return result_history


    def option_price_iteration(
        self,
        final_res,
        scoring_sim_data_val,
        expiry_val,
        rate_val,
        strike_val,
    ):
        option_price_iteration = {}
        stoptime_rl_iteration = {}
        for k, v in final_res.items():
            option_price_iteration[k], stoptime_rl_iteration[k] = option_price(
                scoring_data=scoring_sim_data_val,
                func=v,
                expiry=expiry_val,
                rate=rate_val,
                strike=strike_val,
            )
        return option_price_iteration, stoptime_rl_iteration

    def run(self):
        final_res = self.fitted_lspi_put_option_iteration(
            expiry=self.expiry_val,
            num_steps=self.num_steps_lspi,
            num_paths=self.num_training_paths_lspi_val,
            spot_price=self.spot_price_val,
            spot_price_frac=self.spot_price_frac_lspi,
            rate=self.rate_val,
            vol=self.vol_val,
            strike=self.strike_val,
            training_iters=self.training_iters_lspi_val,
        )


        scoring_sim_data_val = scoring_sim_data(
            expiry=self.expiry_val,
            num_steps=self.num_steps_scoring,
            num_paths=self.num_scoring_paths_val,
            spot_price=self.spot_price_val,
            rate=self.rate_val,
            vol=self.vol_val,
        )

        option_price_iteration, stoptime_rl_iteration = self.option_price_iteration(
            final_res,
            scoring_sim_data_val,
            self.expiry_val,
            self.rate_val,
            self.strike_val,
        )

        return option_price_iteration, stoptime_rl_iteration

    def clean_and_visualize_iteration_decision(self, stoptime_rl_iteration):
        df = pd.DataFrame(stoptime_rl_iteration).melt(var_name='IterationNumber', value_name='Values')
        df['Iter'] = pd.Categorical(df['IterationNumber'])

        g = (ggplot(df, aes(x='Values', fill='red'))
             + geom_histogram(position="dodge2", binwidth=0.5, center=0,  show_legend=False)
             + scale_x_continuous(breaks=range(0, 51, 2))
             + facet_grid('Iter ~ .', scales='fixed') +
 #            + labs(title='Histogram of Exercise Decision Times, at each Iteration of Q-Learning') +
             labs(y='Frequency (from total of 5,000 Paths)', x="The Stopping Time Step (k)") +
                theme(axis_text_x=element_text(size=10, color="black",angle=90),
                text = element_text(size = 10,color='black'),
                axis_text_y=element_text(size=10, color="black")))
               # ggtitle("S0 = 36, Strike Price = 40, Volatility = 0.2"))
        return g
    
    def clean_and_visualize_option_price_value(self, option_price_iteration):
        
        df = pd.DataFrame({'IterationNumber': list(option_price_iteration.keys()), 
                           'Values': list(option_price_iteration.values())})

        bar_chart = (ggplot(df, aes(x='IterationNumber', y='Values')) +
             geom_bar(stat='identity', fill='blue', width = 0.5) +
            labs(x='Iteration Number', y='Value of the Option (Calculated from the Policy Derived in the Current Iteration)') +
             theme(text = element_text(size = 14)) +
            scale_y_continuous(expand=(0, 0)))
        
        return bar_chart 


#option_pricing_rl_move = OptionPricing_RL_move(num_scoring_paths_val=50000, 
#                                               num_training_paths_lspi_val=5000)
#price_ite, decision_ite = option_pricing_rl_move.run()

#print(price_ite)

#ite_plot =option_pricing_rl_move.clean_and_visualize_iteration_decision(decision_ite)
#print(ite_plot)

#ggsave(self=ite_plot, filename='Results/Fig/ite_decision_s036_v_0_2.png', dpi = 600)


#price_plot = option_pricing_rl_move.clean_and_visualize_option_price_value(price_ite)

#print(price_plot)


#def clean_and_visualize_option_price_value_sepfunc(price_ite):
#        
#        df = pd.DataFrame({'IterationNumber': list(price_ite.keys()), 
#                           'Values': list(price_ite.values())})

#        bar_chart = (ggplot(df, aes(x='IterationNumber', y='Values')) +
#             geom_bar(stat='identity', fill='blue', width = 0.5) +
#            labs(x='Iteration Number', y='Value of the Option') +
#             theme(text = element_text(size = 10)) +
#            scale_y_continuous(expand=(0, 0)))
        
#        return bar_chart 

#price_plot = clean_and_visualize_option_price_value_sepfunc(price_ite)
#print(price_plot)
#ggsave(self=price_plot, filename='Results/Fig/price_s036_v_0_2.png', dpi = 600)

# I should continue with 5000 paths, 3iterations
# And bring the plots to the paper
# maybe I shoudl update the Table1 results too