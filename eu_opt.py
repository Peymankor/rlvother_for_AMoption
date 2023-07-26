
from scipy.stats import norm
import numpy as np

def european_put_price(
    spot_price: float,
    expiry: float,
    rate: float,
    vol: float,
    strike: float
) -> float:
    sigma_sqrt: float = vol * np.sqrt(expiry)
    d1: float = (np.log(spot_price / strike) +
                 (rate + vol ** 2 / 2.) * expiry) \
        / sigma_sqrt
    d2: float = d1 - sigma_sqrt
    return strike * np.exp(-rate * expiry) * norm.cdf(-d2) \
        - spot_price * norm.cdf(-d1)