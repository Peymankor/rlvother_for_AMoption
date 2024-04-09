
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
class OptimalExerciseLSM_GARCH:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    num_steps: int
    rate: float
    
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

        num_steps = self.num_steps   
        num_paths: int = scoring_data.shape[0]
        prices: np.ndarray = np.zeros(num_paths)
        stoptime: np.ndarray = np.ones(num_paths)*51
        dt: float = self.expiry / self.num_steps

        #Beta_list.reverse()

        for i, path in enumerate(scoring_data):
            step: int = 1
            while step <= num_steps:
                t: float = step * dt
                exercise_price: float = self.payoff(t, path[step-1])
                
                if exercise_price>0:
                    XX=laguerre_polynomials_ind(path[step-1],k)
                    continue_price: float = np.dot(XX, Beta_list[step-1])  \
                        if step < num_steps else 0.

                #continue_price: float = func.evaluate([(t, path[step])])[0] \
                #    if step < self.num_steps else 0.
                    step += 1
                    if exercise_price >= continue_price:
                        prices[i] = np.exp(-self.rate * t) * exercise_price
                        stoptime[i] = step - 1
                        step = num_steps + 1
                        #stoptime[i] = t
                        #stoptime[i] = step-1
                        #print(step-1)
                else:
                    step += 1


        return np.average(prices), stoptime


################# LSM price function ###########################################