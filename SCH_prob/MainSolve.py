import random
from typing import *
import matplotlib.pyplot as plt
from NSGAII import * 

# =========================================================
# NSGA-II Solve for SCH + Plot
# =========================================================
def solve_sch_nsga2(
    pop_size: int = 100,
    generations: int = 200,
    lower: float = -1000.0,
    upper: float = 1000.0,
    eta_c: float = 20.0,
    eta_m: float = 100.0,
    p_c: float = 0.9,
    seed: int | None = 1,
) -> List[List[float]]:
    """
    Run NSGA-II and return final parent population Pt.
    """
    if seed is not None:
        random.seed(seed)

    # P0
    Pt = [[random.uniform(lower, upper)] for _ in range(pop_size)]
    # Q0 (the usual flow generates it from P0)
    Qt = make_new_pop(Pt, evaluate_sch, lower, upper, eta_c, eta_m, p_c)

    for _ in range(generations):
        # Rt = Pt ∪ Qt
        Rt = Pt + Qt
        objs_R = [evaluate_sch(ind) for ind in Rt]

        # sort Rt into fronts
        fronts_R = fast_non_dominated_sort(Rt, evaluate_sch)

        # crowding on Rt (global list)
        crowd_R = [0.0] * len(Rt)
        for Fi in fronts_R:
            crowding_distance_assignment(Fi, objs_R, crowd_R)

        # Build Pt+1 by filling fronts
        Pt1: List[List[float]] = []
        i = 0
        while i < len(fronts_R) and (len(Pt1) + len(fronts_R[i]) <= pop_size):
            Pt1.extend([Rt[idx] for idx in fronts_R[i]])
            i += 1

        # partial fill from next front using crowding distance
        if len(Pt1) < pop_size and i < len(fronts_R):
            Fi = fronts_R[i]
            Fi_sorted = sorted(Fi, key=lambda idx: -crowd_R[idx])
            need = pop_size - len(Pt1)
            Pt1.extend([Rt[idx] for idx in Fi_sorted[:need]])

        Pt = Pt1
        Qt = make_new_pop(Pt, evaluate_sch, lower, upper, eta_c, eta_m, p_c)

    return Pt


def plot_pareto_fronts_sch(population: List[List[float]], max_fronts: int = 5) -> None:
    objs = [evaluate_sch(ind) for ind in population]
    fronts = fast_non_dominated_sort(population, evaluate_sch)

    k = min(max_fronts, len(fronts))
    plt.figure(figsize=(7, 5))

    for fi in range(k):
        idxs = fronts[fi]
        xs = [objs[i][0] for i in idxs]
        ys = [objs[i][1] for i in idxs]
        plt.scatter(xs, ys, s=18, label=f"F{fi+1} (n={len(idxs)})")

    plt.xlabel("f1(x) = x^2")
    plt.ylabel("f2(x) = (x-2)^2")
    plt.title("NSGA-II on SCH: Pareto Fronts")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Pt_final = solve_sch_nsga2(
        pop_size=100,
        generations=100,
        seed=1,
        lower=-1000.0,
        upper=1000.0,
        eta_c=20.0,
        eta_m=100.0,
        p_c=0.9
    )
    plot_pareto_fronts_sch(Pt_final, max_fronts=5)

