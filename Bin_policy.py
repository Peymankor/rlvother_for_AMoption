
from dataclasses import dataclass
from typing import Tuple, Sequence
import numpy as np

@dataclass(frozen=True)
class BinPolicy:
        
    """ 
    This is Binomial lattice class, where th first function fidn the price and second 
    gives the continuation valaue at each time. The underlying price is GBM. 
    """
    spot_price: float
    strike: float
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def option_price(self) -> float:
        
        """
        This is the function to find price of option using Binomial lattice.
        """

        # Number of time steps
        dt = self.expiry/self.num_steps
    
        # u value 
        u =  np.exp(self.vol * np.sqrt(dt))
    
        # 1/u value
        d = 1/u
    
        # probability of q 
        q = (np.exp(self.rate * dt) - d)/(u - d)
    
        # probability of q
    
        # 
        C = {}
        #price_array = []
        
        #future_value_list = []
    
        #payoff function for put options    
        for m in range(0, self.num_steps+1):

            C[(self.num_steps, m)] = max(self.strike - 
                self.spot_price * (u ** (2*m - self.num_steps)), 0) 
    
    
        for k in range(self.num_steps-1, -1, -1):

            for m in range(0,k+1):

                future_value = np.exp(-self.rate * dt) * (q * C[(k+1, m+1)] + 
                                (1-q) * C[(k+1, m)])
                exercise_value =  max(self.strike - self.spot_price * (u ** (2*m-k)),0)
            
                C[(k, m)] = max(future_value, exercise_value)
    
        return C[(0,0)]


    def continuation_value_list(self) -> Sequence[Tuple[int,float,float]]:

          
        """
        This function will give output with three tuple, first time index, second
        is node price and third constinuation value.
        """

        # Number of time steps
        dt = self.expiry/self.num_steps
    
        # u value 
        u =  np.exp(self.vol * np.sqrt(dt))
    
        # 1/u value
        d = 1/u
    
        # probability of q 
        q = (np.exp(self.rate * dt) - d)/(u - d)
    
        # probability of q
    
        # 
        C = {}
        #price_array = []
        
        future_value_list = []
    
        #payoff function for put options    
        for m in range(0, self.num_steps+1):

            C[(self.num_steps, m)] = max(self.strike - 
                self.spot_price * (u ** (2*m - self.num_steps)), 0) 
    
    
        for k in range(self.num_steps-1, -1, -1):

            for m in range(0,k+1):
                
                node_price = self.spot_price * (u ** (2*m-k))

                exercise_value =  max(self.strike - node_price,0)

                future_value = np.exp(-self.rate * dt) * (q * C[(k+1, m+1)] + (1-q) * C[(k+1, m)])
            
                future_value_list.append((k,node_price,future_value))

                C[(k, m)] = max(future_value, exercise_value)
        
        return future_value_list


if __name__ == "__main__":
    
    print("Test Binomial lattice code:\n")

    S0_value = 36
    r_value = 0.06
    sd_value = 0.2
    T_value = 1
    steps_value = 50
    strike = 40

    Binpolicy_calss = BinPolicy(spot_price=S0_value, strike=strike,
                expiry=T_value, rate=r_value, vol=sd_value, 
                num_steps=steps_value)

    print("The Option Price:")

    print(Binpolicy_calss.option_price())