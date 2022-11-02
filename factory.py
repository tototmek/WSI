from math import exp

RESOURCES = [
    [20,    (1, 1)],
    [10,    (-0.5, 1)],
    [5,     (-1, -0.5)],
    [10,    (1, -1)],
]


def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[0])

# Funkcja celu:


def q(x):
    cost = 0
    for resource in RESOURCES:
        cost += resource[0] * (1 - exp(-manhattan_dist(x, resource[1])))
    return cost
