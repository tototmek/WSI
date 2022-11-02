from factory import q
from evolution import evolution
import random

POP_SIZE = 1000
population = []


for i in range(POP_SIZE):
    population.append((random.uniform(-4, -4), random.uniform(-4, -4)))


pop, scores = evolution(q, population, len(population), 0.1, 0, 1000)
for i in range(len(pop)):
    print(str(pop[i]) + ", " + str(scores[i]))
