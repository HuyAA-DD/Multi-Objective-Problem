"""Minimal MOEA/D implementation for a 1D decision variable.

Problem:
  f1(x) = x^2
  f2(x) = (x-2)^2
  x in [0, 2]
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Tuple
import matplotlib.pyplot as plt


@dataclass
class Individual:
    x: float
    f1: float
    f2: float


def evaluate(x: float) -> Tuple[float, float]:
    """Compute objective values for x."""
    return x * x, (x - 2.0) * (x - 2.0)


def clamp(x: float, lo: float = 0.0, hi: float = 2.0) -> float:
    return max(lo, min(hi, x))


def init_population(n: int, rng: random.Random) -> List[Individual]:
    pop = []
    for _ in range(n):
        x = rng.uniform(0.0, 2.0)
        f1, f2 = evaluate(x)
        pop.append(Individual(x, f1, f2))
    return pop


def generate_weights(n: int) -> List[Tuple[float, float]]:
    """Generate N evenly spaced 2D weight vectors on the unit simplex."""
    if n < 2:
        return [(0.5, 0.5)]
    return [(i / (n - 1), 1.0 - i / (n - 1)) for i in range(n)]


def compute_neighbors(weights: List[Tuple[float, float]], t: int) -> List[List[int]]:
    """Find T nearest neighbors for each weight (by Euclidean distance)."""
    neighbors = []
    for i, w in enumerate(weights):
        dists = []
        for j, wj in enumerate(weights):
            d = math.dist(w, wj)
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        neighbors.append([idx for _, idx in dists[:t]])
    return neighbors


def tchebycheff(ind: Individual, weight: Tuple[float, float], z: Tuple[float, float]) -> float:
    """Tchebycheff scalarization with ideal point z."""
    return max(
        weight[0] * abs(ind.f1 - z[0]),
        weight[1] * abs(ind.f2 - z[1]),
    )


def differential_mutation(x1: float, x2: float, x3: float, f: float, rng: random.Random) -> float:
    return clamp(x1 + f * (x2 - x3))


def polynomial_mutation(x: float, eta: float, rng: random.Random) -> float:
    """Polynomial mutation for 1D, bounded."""
    u = rng.random()
    if u < 0.5:
        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
    else:
        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
    return clamp(x + delta * 2.0)  # scale by domain size


def moead(
    pop_size: int = 51,
    n_gen: int = 200,
    t_neighbors: int = 10,
    f_scale: float = 0.5,
    eta_mut: float = 20.0,
    seed: int = 42,
) -> List[Individual]:
    rng = random.Random(seed)
    weights = generate_weights(pop_size)
    neighbors = compute_neighbors(weights, t_neighbors)
    pop = init_population(pop_size, rng)

    # Ideal point
    z = (
        min(ind.f1 for ind in pop),
        min(ind.f2 for ind in pop),
    )

    for _ in range(n_gen):
        for i in range(pop_size):
            # pick 3 neighbors
            nb = neighbors[i]
            r1, r2, r3 = rng.sample(nb, 3)
            x1, x2, x3 = pop[r1].x, pop[r2].x, pop[r3].x

            # variation: DE + polynomial mutation
            y = differential_mutation(x1, x2, x3, f_scale, rng)
            if rng.random() < 0.3:
                y = polynomial_mutation(y, eta_mut, rng)

            f1, f2 = evaluate(y)
            child = Individual(y, f1, f2)

            # update ideal point
            z = (min(z[0], child.f1), min(z[1], child.f2))

            # update neighbors by Tchebycheff
            for j in nb:
                if tchebycheff(child, weights[j], z) <= tchebycheff(pop[j], weights[j], z):
                    pop[j] = child

    return pop


def pareto_front(pop: List[Individual]) -> List[Individual]:
    """Extract non-dominated solutions."""
    front = []
    for i, a in enumerate(pop):
        dominated = False
        for j, b in enumerate(pop):
            if i == j:
                continue
            if (b.f1 <= a.f1 and b.f2 <= a.f2) and (b.f1 < a.f1 or b.f2 < a.f2):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def main() -> None:
    pop = moead()
    front = sorted(pareto_front(pop), key=lambda ind: ind.x)
    
    # Extract f1 and f2 values
    f1_values = [ind.f1 for ind in front]
    f2_values = [ind.f2 for ind in front]
    
    # Plot Pareto front
    plt.figure(figsize=(8, 6))
    plt.scatter(f1_values, f2_values, color='blue', s=50, label='Pareto Front')
    plt.xlabel('f1(x) = x²', fontsize=12)
    plt.ylabel('f2(x) = (x-2)²', fontsize=12)
    plt.title('Pareto Front - MOEA/D', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
