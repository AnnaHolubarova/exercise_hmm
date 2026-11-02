# Task 1
import json
import itertools
import math


def log2(p):
    """Compute the base-2 logarithm of a probability, handling zero."""
    if p == 0:
        return -math.inf
    else:
        return math.log2(p)


def read_model(path):
    """Read a JSON model file and return its HMM."""
    with open(path, "r") as model_file:
        model = json.load(model_file)
    return HMM(**model)


def log_sum(a, b):
    """Compute the base-2 logarithm of the sum of two probabilities in log space."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + log2(1 + 2 ** (-abs(a - b)))


def agreement(z1, z2):
    """Compute the agreement between two sequences of states."""
    if len(z1) != len(z2):
        raise ValueError("Sequences must be of the same length.")
    return sum(1 for a, b in zip(z1, z2) if a == b)


class HMM:
    """A class representing a Hidden Markov Model."""

    def __init__(self, states, alphabet, pi, a, e):
        state_keys = set(states)
        alphabet_keys = set(alphabet)

        if set(pi) != state_keys:
            raise ValueError("Initial probabilities must have exactly the model states as keys.")
        if set(a) != state_keys:
            raise ValueError("Transition probabilities must have exactly the model states as rows.")
        if set(e) != state_keys:
            raise ValueError("Emission probabilities must have exactly the model states as rows.")
        if any(set(row) != state_keys for row in a.values()):
            raise ValueError("Each transition row must have exactly the model states as keys.")
        if any(set(row) != alphabet_keys for row in e.values()):
            raise ValueError("Each emission row must have exactly the alphabet symbols as keys.")

        if not math.isclose(sum(pi.values()), 1):
            raise ValueError("Initial probabilities must sum to 1.")
        if any(not math.isclose(sum(row.values()), 1) for row in a.values()):
            raise ValueError("Each transition row must sum to 1.")
        if any(not math.isclose(sum(row.values()), 1) for row in e.values()):
            raise ValueError("Each emission row must sum to 1.")

        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e
        self.log_pi = {state: log2(probability) for state, probability in pi.items()}
        self.log_a = {
            source: {target: log2(probability) for target, probability in row.items()}
            for source, row in a.items()
        }
        self.log_e = {
            state: {symbol: log2(probability) for symbol, probability in row.items()}
            for state, row in e.items()
        }

    def joint_probability(self, x, z):
        """Return the joint probability of an observation sequence and state path."""
        if len(x) != len(z):
            raise ValueError("Sequence and path must be of the same length.")

        probability = 1.0
        previous_state = None
        for index, (symbol, state) in enumerate(zip(x, z)):
            if index == 0:
                probability *= self.pi[state]
            else:
                probability *= self.a[previous_state][state]
            probability *= self.e[state][symbol]
            previous_state = state
        return probability

    def log_joint(self, x, z):
        """Return the log joint probability of an observation sequence and state path."""
        if len(x) != len(z):
            raise ValueError("Sequence and path must be of the same length.")

        log_probability = 0.0
        previous_state = None
        for index, (symbol, state) in enumerate(zip(x, z)):
            if index == 0:
                log_probability += self.log_pi[state]
            else:
                log_probability += self.log_a[previous_state][state]
            log_probability += self.log_e[state][symbol]
            previous_state = state
        return log_probability

    def brute_force(self, x):
        """Return the most likely path, its probability, and total probability."""
        best_path = None
        best_probability = -1.0
        total_probability = 0.0

        for path in itertools.product(self.states, repeat=len(x)):
            path_string = "".join(path)
            probability = self.joint_probability(x, path_string)
            total_probability += probability
            if probability > best_probability:
                best_path = path_string
                best_probability = probability

        return best_path, best_probability, total_probability

#print(read_model("weather_model.json").joint_probability("sr", "HL"))
#print(read_model("weather_model.json").brute_force("ssrssrrrss"))
