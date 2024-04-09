
## Import packages for Bin
from typing import Callable, Dict, Sequence, Tuple, List

from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal

#######################################################

def bin_tree_price(spot_price_val, strike_val,expiry_val, rate_val,
                    vol_val):

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
            spot_price=spot_price_val,
            payoff=lambda _, x: max(strike_val - x, 0),
            expiry=expiry_val,
            rate=rate_val,
            vol=vol_val,
            num_steps=100
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    bin_tree_price: float = vf_seq[0][NonTerminal(0)]
    bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
            opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
    bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)

    return bin_tree_price, bin_tree_x, bin_tree_y