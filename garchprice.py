## This is the code to generate the GARCH price model,
## Using the saved parameters of teh GRACH model

import numpy as np
import scipy.stats as ss

class BrentGARCHPrice_Light:
    
    def generate_paths(self, num_time_steps, num_path_numbers, 
                       initial_price, expiry_val):
        # Pre-saved GARCH parameters, to Nov 2023 date

        mu = 0.0619
        omega = 0.0657
        alpha = 0.0881
        beta = 0.901


        #mu = 0.0608
        #omega = 0.064
        #alpha = 0.087
        #beta = 0.901

        num_time_steps_per_day  = expiry_val*360

        resid_sim = np.zeros((num_path_numbers, num_time_steps_per_day+1))
        conditional_sim = np.zeros((num_path_numbers, num_time_steps_per_day+1))
        S = np.zeros((num_path_numbers, num_time_steps_per_day +1))
        S[:, 0] = initial_price

        for path in range(num_path_numbers):
            for t in range(1, num_time_steps_per_day):
                conditional_sim[path, t] = (omega + alpha * resid_sim[path, t-1]**2 + beta * conditional_sim[path, t-1]**2)**(1/2)
                r = ss.norm.rvs(loc=mu, scale=conditional_sim[path, t])
                resid_sim[path, t] = r - 0
                S[path, t] = S[path, t-1] * (1 + (r / 100))
        garch_price_all_index = S[:, 1:]
        
        step_size = expiry_val*360 // num_time_steps
        indexes = [i * step_size for i in range(num_time_steps)]
        #print("Indexes: ", indexes)
        garch_prices_indexed = garch_price_all_index[:, indexes]
        
        
        return garch_prices_indexed



GARCH_Class = BrentGARCHPrice()

prices = GARCH_Class.generate_paths(num_time_steps=50, num_path_numbers=100, 
                                    initial_price=80, expiry_val=1)
prices
#prices.shape

import matplotlib.pyplot as plt
for row in prices:
    plt.plot(row)

plt.show()

