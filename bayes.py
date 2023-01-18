import random

probability_table = {
    "T": [
        (None, 0.02)
        ],
    "W": [
        (None, 0.01)
        ],
    "A": [
        ([("W", True), ("T", True)], 0.95),
        ([("W", True), ("T", False)], 0.94),
        ([("W", False), ("T", True)], 0.29),
        ([("W", False), ("T", False)], 0.001)
        ],
    "J": [
        ([("A", True)], 0.90),
        ([("A", False)], 0.05)
    ],
    "M": [
        ([("A", True)], 0.70),
        ([("A", False)], 0.01)
    ],
}

class BayesNet:
    def __init__(self, probability_table):
        self.probability_table = probability_table

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

    def get_probalility_node_to_node(self, from_node, from_value, to_node, to_value):
        for row in self.probability_table[to_node]:
            if row[0] is None:
                continue
            for parent, value in row[0]:
                if parent == from_node and value == from_value:
                    return row[1] if to_value else 1 - row[1]
        return 0
    
    def get_parental_probability(self, node, state):
        for row in self.probability_table[node]:
            if row[0] is None:
                return row[1]
            valid = True
            for parent, value in row[0]:
                if state[parent] != value:
                    valid = False
                    break
            if valid:
                return row[1]
        return 0

    def sample(self, node, state):
        p = self.get_parental_probability(node, state)

            
        return random.random() < p
        
    def mcmc(self, evidence, query, iterations):

        # Initialize the state of the network
        state = {}
        for node in self.probability_table:
            state[node] = None

        # Set the evidence
        for node, value in evidence.items():
            state[node] = value

        # Initialize the counts
        counts = {}
        for node in self.probability_table:
            counts[node] = 0

        # Run the MCMC algorithm
        for i in range(iterations):
            node = self.get_random_node()
            if node not in evidence:
                state[node] = self.sample(node, state)
            if state[query]:
                counts[query] += 1

        return counts[query] / iterations




if __name__ == "__main__":

    # Create the Bayes net
    bayes_net = BayesNet(probability_table)


    evidence = {"J": True, "M": True}
    query = "A"
    print(f"P({query}|{evidence}) = ", bayes_net.mcmc(evidence, query, 100000))

    evidence = {}
    query = "W"
    print(f"P({query}|{evidence}) = ", bayes_net.mcmc(evidence, query, 100000))

    evidence = {"T": True, "W": False}
    query = "A"
    print(f"P({query}|{evidence}) = ", bayes_net.mcmc(evidence, query, 100000))

    evidence = {"A": True}
    query = "J"
    print(f"P({query}|{evidence}) = ", bayes_net.mcmc(evidence, query, 100000))

