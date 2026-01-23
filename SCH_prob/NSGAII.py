from typing import *
from SCH import *
import random
# =========================================================
# Pareto dominance
# =========================================================
def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """
    a dominates b (minimization):
    a không tệ hơn b ở mọi mục tiêu và tốt hơn ít nhất 1 mục tiêu.
    """
    better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            better = True
    return better


# =========================================================
# Fast non-dominated sorting (Deb)
# =========================================================
def fast_non_dominated_sort(
    population: List[List[float]],
    evaluate_all: Callable[[List[float]], Tuple[float, ...]]
) -> List[List[int]]:
    """
    Returns fronts = [F1, F2, ...], each Fi is a list of indices in population.
    """
    N = len(population)
    if N == 0:
        return []

    objs: List[Tuple[float, ...]] = [evaluate_all(ind) for ind in population]

    S: List[List[int]] = [[] for _ in range(N)]  # S[p] = set of solutions dominated by p
    n: List[int] = [0] * N                       # n[p] = number of solutions dominating p
    fronts: List[List[int]] = [[]]               # F1

    # Find first front
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    # Subsequent fronts
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def build_rank_from_fronts(fronts: List[List[int]], N: int) -> List[int]:
    rank = [0] * N
    for f_idx, front in enumerate(fronts, start=1):
        for i in front:
            rank[i] = f_idx
    return rank


# =========================================================
# Crowding distance assignment (in-place)
# =========================================================
def crowding_distance_assignment(
    front: List[int],
    objs: List[Tuple[float, ...]],
    crowding: List[float],
) -> None:
    """
    Update crowding distance for indices in 'front' (NSGA-II).
    """
    k = len(front)
    if k == 0:
        return

    M = len(objs[front[0]])

    if k <= 2:
        for idx in front:
            crowding[idx] = float("inf")
        return

    # reset for this front
    for idx in front:
        crowding[idx] = 0.0

    for m in range(M):
        front_sorted = sorted(front, key=lambda i: objs[i][m])
        fmin = objs[front_sorted[0]][m]
        fmax = objs[front_sorted[-1]][m]

        crowding[front_sorted[0]] = float("inf")
        crowding[front_sorted[-1]] = float("inf")

        if fmax == fmin:
            continue

        for j in range(1, k - 1):
            idx = front_sorted[j]
            if crowding[idx] == float("inf"):
                continue
            prev_idx = front_sorted[j - 1]
            next_idx = front_sorted[j + 1]
            crowding[idx] += (objs[next_idx][m] - objs[prev_idx][m]) / (fmax - fmin)


# =========================================================
# Selection: binary tournament with crowded-comparison
# =========================================================
def tournament_select_index(pop_size: int, rank: List[int], crowding: List[float]) -> int:
    a = random.randrange(pop_size)
    b = random.randrange(pop_size)

    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b

    if crowding[a] > crowding[b]:
        return a
    if crowding[b] > crowding[a]:
        return b

    return a if random.random() < 0.5 else b

# =========================================================
# Variation operators: SBX + Polynomial Mutation (Deb)
# =========================================================
def sbx_crossover(
    parent1: List[float],
    parent2: List[float],
    eta_c: float = 20.0,
    p_c: float = 0.9,
    lower: float = -1000.0,
    upper: float = 1000.0,
) -> Tuple[List[float], List[float]]:
    """
    SBX crossover (real-coded).
    """
    if random.random() > p_c:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    c1 = parent1.copy()
    c2 = parent2.copy()
    EPS = 1e-14

    for i in range(n):
        x1, x2 = parent1[i], parent2[i]
        if abs(x1 - x2) < EPS:
            continue

        if x1 > x2:
            x1, x2 = x2, x1

        lo, hi = lower, upper
        if hi <= lo:
            raise ValueError("Invalid bounds (upper <= lower)")

        # gene-wise crossover prob 0.5 (common implementation)
        if random.random() > 0.5:
            continue

        u = random.random()

        beta = 1.0 + (2.0 * (x1 - lo) / (x2 - x1))
        alpha = 2.0 - (beta ** (-(eta_c + 1.0)))
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (hi - x2) / (x2 - x1))
        alpha = 2.0 - (beta ** (-(eta_c + 1.0)))
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        child1 = min(max(child1, lo), hi)
        child2 = min(max(child2, lo), hi)

        if random.random() <= 0.5:
            c1[i], c2[i] = child2, child1
        else:
            c1[i], c2[i] = child1, child2

    return c1, c2


def polynomial_mutation(
    individual: List[float],
    eta_m: float = 100.0,
    p_m: float | None = None,
    lower: float = -1000.0,
    upper: float = 1000.0,
) -> List[float]:
    """
    Polynomial mutation (Deb).
    """
    n = len(individual)
    if n == 0:
        return []
    if p_m is None:
        p_m = 1.0 / n

    mutated = individual.copy()
    EPS = 1e-14

    for i in range(n):
        if random.random() > p_m:
            continue

        x = mutated[i]
        lo, hi = lower, upper
        if hi <= lo:
            raise ValueError("Invalid bounds (upper <= lower)")

        x = min(max(x, lo), hi)
        if (hi - lo) < EPS:
            mutated[i] = lo
            continue

        delta1 = (x - lo) / (hi - lo)
        delta2 = (hi - x) / (hi - lo)
        r = random.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if r < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta_m + 1.0))
            delta_q = (val ** mut_pow) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta_m + 1.0))
            delta_q = 1.0 - (val ** mut_pow)

        x_new = x + delta_q * (hi - lo)
        mutated[i] = min(max(x_new, lo), hi)

    return mutated


# =========================================================
# make-new-pop(P): generate offspring Q
# =========================================================
def make_new_pop(
    P: List[List[float]],
    evaluate_all: Callable[[List[float]], Tuple[float, ...]],
    lower: float,
    upper: float,
    eta_c: float,
    eta_m: float,
    p_c: float,
) -> List[List[float]]:
    """
    Create offspring population Q from parent population P:
    tournament + SBX + polynomial mutation.
    """
    N = len(P)
    fronts = fast_non_dominated_sort(P, evaluate_all)
    rank = build_rank_from_fronts(fronts, N)

    objs = [evaluate_all(ind) for ind in P]
    crowding = [0.0] * N
    for Fi in fronts:
        crowding_distance_assignment(Fi, objs, crowding)

    Q: List[List[float]] = []
    while len(Q) < N:
        i1 = tournament_select_index(N, rank, crowding)
        i2 = tournament_select_index(N, rank, crowding)

        c1, c2 = sbx_crossover(P[i1], P[i2], eta_c=eta_c, p_c=p_c, lower=lower, upper=upper)
        c1 = polynomial_mutation(c1, eta_m=eta_m, p_m=None, lower=lower, upper=upper)
        c2 = polynomial_mutation(c2, eta_m=eta_m, p_m=None, lower=lower, upper=upper)

        Q.append(c1)
        if len(Q) < N:
            Q.append(c2)

    return Q