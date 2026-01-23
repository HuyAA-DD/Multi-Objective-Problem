# =========================================================
# 1) Problem: SCH (Schaffer)
# =========================================================
from typing import *
def evaluate_sch(ind: List[float]) -> Tuple[float, float]:
    """
    SCH objectives (minimization):
      f1(x) = x^2
      f2(x) = (x - 2)^2
    individual: [x]
    """
    x = ind[0]
    return (x * x, (x - 2.0) * (x - 2.0))
