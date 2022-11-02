import numpy as np
import random

MUTATION_STRENGTH = 0.4


def mutate(x):
    mutations = np.random.normal(0, MUTATION_STRENGTH, 2)
    return (x[0] + mutations[0], x[1] + mutations[1])


def crossover(x1, x2):
    return (
        (x1[0], x2[1]),
        (x2[0], x1[1]))


def evaluate(cost_fun, P):
    scores = [0 for _ in range(len(P))]
    for i, x in enumerate(P):
        scores[i] = cost_fun(x)
    return scores


def apply_mutations_and_crossovers(P, cross_chance, mut_chance):
    for i in range(len(P)):
        if random.uniform(0, 1) < cross_chance:
            P[i] = crossover(P[i], random.choice(P))[0]
        if random.uniform(0, 1) < mut_chance:
            P[i] = mutate(P[i])


def scores_to_probabilities(scores, metric="cost"):
    # Zamiana metryki typu "cost" na "fitness"
    if metric == "cost":
        max_score = max(scores)
        scores = [max_score - score for score in scores]
    # Normalizacja
    total_scores = sum(scores)
    if total_scores == 0:
        return [1 / len(scores) for i in range(len(scores))]
    return [score / total_scores for score in scores]


def reproduce(P, s):
    new_population = []
    probabilities = scores_to_probabilities(s)
    indices = np.random.choice(range(len(P)), p=probabilities, size=len(P))
    for index in indices:
        new_population.append(P[index])
    return new_population


def evolution(cost_fun, P, m, mutate_chance, cross_chance, total_generations):
    if len(P) != m:
        raise ValueError("Początkowa populacja musi mieć m osobników!")
    for i in range(total_generations):
        score = evaluate(cost_fun, P)
        P = reproduce(P, score)
        apply_mutations_and_crossovers(P, cross_chance, mutate_chance)
    scores = evaluate(cost_fun, P)
    return P, scores
