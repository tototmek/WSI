probability_table = {
    "T": [
        (None, 0.02)
        ],
    "W": [
        (None, 0.01)
        ],
    "A": [
        ([("W", True), ("T", True)], 0.95)
        ([("W", True), ("T", False)], 0.94)
        ([("W", False), ("T", True)], 0.29)
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
            for node in self.probability_table:
                if state[node] is None:
                    state[node] = self.sample(node, state)
            if state[query]:
                counts[query] += 1

        return counts[query] / iterations

    def sample(self, node, state):
        # Get the probability table for the node
        probability_table = self.probability_table[node]

        # Get the parents of the node
        parents = []
        for parent, value in probability_table[0][0]:
            parents.append(parent)

        # Get the probabilities for the node
        probabilities = []
        for parent_values, probability in probability_table:
            if parent_values is None:
                probabilities.append(probability)
            else:
                match = True
                for parent, value in parent_values:
                    if state[parent] != value:
                        match = False
                if match:
                    probabilities.append(probability)

        # Sample from the probabilities
        r = random.random()
        for i in range(len(probabilities)):
            if r < probabilities[i]:
                return True
            else:
                r -= probabilities[i]
        return False

if __name__ == "__main__":

    # Create the Bayes net
    bayes_net = BayesNet(probability_table)

    # Run the MCMC algorithm
    evidence = {"W": True}
    query = "J"
    iterations = 10000
    probability = bayes_net.mcmc(evidence, query, iterations)
    print(f"P({query}|{evidence}) = {probability}")

