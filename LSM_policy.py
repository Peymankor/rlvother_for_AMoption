
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Callable, Dict, Sequence, Tuple, List

from scipy.stats import norm
from rich import print, pretty
pretty.install()

from price_model import SimulateGBM
from basis_fun import laguerre_polynomials, laguerre_polynomials_ind


@dataclass(frozen=True)
class OptimalExerciseLSM:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int


    def european_put_price(self, strike) -> float:
  
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
  
        return strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) \
            - self.spot_price * norm.cdf(-d1)

    def GBMprice_training(self, num_paths_train, seed_random = True) -> np.ndarray:
        
        return SimulateGBM(
            S0=self.spot_price,
            r=self.rate, sd=self.vol, T=self.expiry, 
            paths=num_paths_train, steps=self.num_steps, 
            seed_random = seed_random)

    
    def train_LSM(self, training_data: np.ndarray ,num_paths_train: int, K:float, k:int):

        steps = int(self.num_steps)
        Stn = training_data
        #Stn = Stock_Matrix
        dt = self.expiry/steps
        cashFlow = np.zeros((num_paths_train, steps))
        cashFlow[:,steps - 1] = np.maximum(K-Stn[:,steps - 1], 0)
      
        cont_value = cashFlow

        decision = np.zeros((num_paths_train, steps))
        decision[:, steps - 1] = 1
        Beta_list = {}
  #discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
  #                                    steps + 1, 1)), paths).reshape((paths, steps))
  
        for index, i in enumerate(reversed(range(steps-1))):

          # Find in the money paths
          in_the_money_n = np.where(K-Stn[:, i] > 0)[0]
          out_of_money_n = np.asarray(list(set(np.arange(num_paths_train)) - set(in_the_money_n)))
          

          X = laguerre_polynomials(Stn[in_the_money_n, i], k)
          Y = cashFlow[in_the_money_n, i + 1]/np.exp(self.rate*dt)

          A = np.dot(X.T, X)
          b = np.dot(X.T, Y)
          Beta = np.dot(np.linalg.pinv(A), b)

          cont_value[in_the_money_n,i] =  np.dot(X, Beta)
    # Yasaman, lokk at below three codes, these are one under discussion :)
          try:
              cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(self.rate*dt)
          except:
              pass

          decision[:, i] = np.where(np.maximum(K-Stn[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
          cashFlow[:, i] =  np.maximum(K-Stn[:, i], cont_value[:,i])
          
          Beta_list.update({i: Beta}) 
            
        return Beta_list

    def scoring_sim_data(self, num_paths_test: int) -> np.ndarray:
            return SimulateGBM(
            S0=self.spot_price,
            r=self.rate, sd=self.vol, T=self.expiry, 
            paths=num_paths_test, steps=self.num_steps)

    def option_price(self,
        scoring_data: np.ndarray,
        Beta_list,
        k
        #func: FunctionApprox[Tuple[float, float]]
    ) -> float:

        num_steps = self.num_steps-1   
        num_paths: int = scoring_data.shape[0]
        prices: np.ndarray = np.zeros(num_paths)
        stoptime: np.ndarray = np.zeros(num_paths)
        dt: float = self.expiry / self.num_steps

        #Beta_list.reverse()

        for i, path in enumerate(scoring_data):
            step: int = 0
            while step <= num_steps:
                t: float = (step+1) * dt
                exercise_price: float = self.payoff(t, path[step])
                
                if exercise_price>0:
                    XX=laguerre_polynomials_ind(path[step],k)
                    continue_price: float = np.dot(XX, Beta_list[step])  \
                        if step < num_steps else 0.

                #continue_price: float = func.evaluate([(t, path[step])])[0] \
                #    if step < self.num_steps else 0.
                    step += 1
                    if exercise_price >= continue_price:
                        prices[i] = np.exp(-self.rate * t) * exercise_price
                        stoptime[i] = step
                        step = num_steps + 1
                        #stoptime[i] = t
                        #stoptime[i] = step-1
                        #print(step-1)
                else:
                    step += 1


        return np.average(prices), stoptime


################# LSM price function ###########################################

def lsm_price(spot_price_val, strike_val, expiry_val, rate_val, 
              vol_val, steps_value,K_value, k_value, 
              num_paths_train_val, num_paths_test_val):
    
    def payoff_func(_: float, s: float) -> float:
            return max(strike_val - s, 0.)
    
    Testclass = OptimalExerciseLSM(spot_price=spot_price_val, payoff=payoff_func,expiry=expiry_val,
                                        rate=rate_val, vol=vol_val,num_steps=steps_value)

    train_data_v = Testclass.GBMprice_training(num_paths_train=num_paths_train_val)

    lsm_policy_v = Testclass.train_LSM(training_data=train_data_v, num_paths_train=num_paths_train_val,
                                            K=K_value, k=k_value)

    test_data_v = Testclass.scoring_sim_data(num_paths_test=num_paths_test_val)

    Option_price,stop_times_val = Testclass.option_price(scoring_data=test_data_v,Beta_list=lsm_policy_v,
                                            k=k_value)
    
    return Option_price, stop_times_val


################################ TEST Code ###################################


if __name__ == "__main__":

    S0_value = 36
    r_value = 0.06
    sd_value = 0.2
    T_value = 1
    paths_value = 1000
    steps_value = 50

    K_value = 40
    k_value = 4

    strike = 40

    def payoff_func(_: float, s: float) -> float:
            return max(strike - s, 0.)

    Testclass = OptimalExerciseLSM(spot_price=S0_value, payoff=payoff_func,expiry=T_value,
                                        rate=r_value, vol=sd_value,num_steps=steps_value)

    train_data_v = Testclass.GBMprice_training(num_paths_train=paths_value)

    lsm_policy_v = Testclass.train_LSM(training_data=train_data_v, num_paths_train=paths_value,
                                            K=K_value, k=k_value)

    test_data_v = Testclass.scoring_sim_data(num_paths_test=10000)

    Option_price,_ = Testclass.option_price(scoring_data=test_data_v,Beta_list=lsm_policy_v,
                                            k=k_value)
    print("Test Code: \nOption Price:")
    print(Option_price)



########################################################################################
########################################################################################
########################################################################################
########################################################################################

    S0_values_table1 = np.arange(36,46, 2)
    sd_values_table1 = np.array([0.2, 0.4])
    T_values_table1 = np.array([1, 2])



    def Table1_func(S0_values,sd_values,T_values, paths_number_train, paths_number_test):
        print("%-10s %-10s %-10s %-20s %-20s %-20s" 
            %("S0","vol", "T", "Closed Form European", "Simulated American", "Early exercise"))

        for S0_table1 in S0_values:
            for sd_table1 in sd_values:
                for T_table1 in T_values:

                    LSMclass = OptimalExerciseLSM(spot_price=S0_table1, payoff=payoff_func,
                        expiry=T_table1, rate=r_value, vol=sd_table1,num_steps=steps_value)

                    euoption = LSMclass.european_put_price(strike=40)

                    train_data_v = LSMclass.GBMprice_training(num_paths_train=paths_number_train)

                    lsm_policy_v = LSMclass.train_LSM(training_data=train_data_v, 
                            num_paths_train=paths_number_train, K=K_value, k=k_value)

                    test_data_v = LSMclass.scoring_sim_data(num_paths_test=paths_number_test)

                    Option_price, _ = LSMclass.option_price(scoring_data=test_data_v, 
                        Beta_list=lsm_policy_v,k=k_value)
      
                    print("%d %10.2f %10d %20.3f %20.3f %20.3f" 
                        %(S0_table1,sd_table1, T_table1, euoption, 
                            Option_price,Option_price-euoption))


    Table1_func(S0_values=S0_values_table1, sd_values=sd_values_table1, 
                    T_values=T_values_table1, paths_number_train=100000, 
                    paths_number_test=1000)