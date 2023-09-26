import numpy as np


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

