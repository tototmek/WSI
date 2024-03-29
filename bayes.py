import random
import time

class BayesNet:
    def __init__(self, probability_table):
        self.probability_table = probability_table
        self.history = {}

    def get_random_node(self):
        return random.choice(list(self.probability_table.keys()))

    def get_node_parents(self, node):
        parents = []
        for row in self.probability_table[node]:
            if row[0] is None:
                continue
            for parent, value in row[0]:
                if parent not in parents:
                    parents.append(parent)
        return parents
    
    # Returns a list of all the children of a node
    def get_node_children(self, node):
        children = []
        for child in self.probability_table:
            for row in self.probability_table[child]:
                if row[0] is None:
                    continue
                for parent, value in row[0]:
                    if parent == node and child not in children:
                        children.append(child)
        return children
    
    # Returns the probability of a node given its parents
    def get_parental_probability(self, node, node_value, state):
        result = 1
        for row in self.probability_table[node]:
            if row[0] is None:
                result = row[1]
                break
            valid = True
            for parent, value in row[0]:
                if state[parent] != value:
                    valid = False
                    break
            if valid:
                result = row[1]
                break
        return result if node_value else 1 - result

    def mcmc(self, evidence, query, iterations):

        self.history = {}
        self.history["time"] = []
        self.history["probability"] = []

        start_time = time.time()

        # Initialize the state of the network
        state = {}
        for node in self.probability_table:
            state[node] = random.choice([True, False])

        # Set the evidence
        for node, value in evidence.items():
            state[node] = value

        # Initialize the counts
        counts = {}
        for node in self.probability_table:
            counts[node] = 0

        # Run the MCMC algorithm
        for i in range(iterations):
            # node = self.get_random_node()
            # if node not in evidence:
            #     state[node] = self.sample(node, state)
            for node in self.probability_table:
                if node in evidence:
                    continue
                state[node] = self.sample(node, state)
            if state[query]:
                counts[query] += 1
            self.history["time"].append(time.time() - start_time)
            self.history["probability"].append(counts[query] / (i+1))

        return counts[query] / iterations

    # Gibbs sampling
    def sample(self, node, state):
        new_state = state.copy()
        
        p_true = self.get_parental_probability(node, True, new_state)
        p_false = 1 - p_true

        children = self.get_node_children(node)

        markov_blanket = []
        for child in children:
            markov_blanket.append(child)
            for parent in self.get_node_parents(child):
                if parent == node:
                    continue
                if parent not in markov_blanket:
                    markov_blanket.append(parent)

        new_state[node] = True
        for mb_node in markov_blanket:
            p_true *= self.get_parental_probability(mb_node, new_state[mb_node], new_state)
        
        new_state[node] = False
        for mb_node in markov_blanket:
            p_false *= self.get_parental_probability(mb_node, new_state[mb_node], new_state)

        return p_true / (p_true + p_false) > random.random()
