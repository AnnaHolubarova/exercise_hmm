# Task 1
import argparse
import json
import itertools
import math
from pathlib import Path

from Bio import SeqIO


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

# Task 2

    def viterbi(self, x):
        """Return the most likely path and its log2 probability."""
        if not x:
            return "", 0.0

        v = [
            {state: self.log_pi[state] + self.log_e[state][x[0]] for state in self.states}
        ]
        back = [{}]

        for symbol in x[1:]:
            v_row = {}
            back_row = {}
            for state in self.states:
                best_previous = self.states[0]
                best_score = v[-1][best_previous] + self.log_a[best_previous][state]
                for previous in self.states[1:]:
                    score = v[-1][previous] + self.log_a[previous][state]
                    if score > best_score:
                        best_score = score
                        best_previous = previous
                v_row[state] = best_score + self.log_e[state][symbol]
                back_row[state] = best_previous
            v.append(v_row)
            back.append(back_row)

        best_state = self.states[0]
        best_score = v[-1][best_state]
        for state in self.states[1:]:
            if v[-1][state] > best_score:
                best_state = state
                best_score = v[-1][state]

        path = [best_state]
        for position in range(len(x) - 1, 0, -1):
            path.append(back[position][path[-1]])
        path.reverse()
        return "".join(path), best_score

    def viterbi_plain(self, x):
        """Return the most likely path and probability using plain products."""
        if not x:
            return "", 1.0

        v = [
            {state: self.pi[state] * self.e[state][x[0]] for state in self.states}
        ]
        back = [{}]

        for symbol in x[1:]:
            v_row = {}
            back_row = {}
            for state in self.states:
                best_previous = self.states[0]
                best_probability = v[-1][best_previous] * self.a[best_previous][state]
                for previous in self.states[1:]:
                    probability = v[-1][previous] * self.a[previous][state]
                    if probability > best_probability:
                        best_probability = probability
                        best_previous = previous
                v_row[state] = best_probability * self.e[state][symbol]
                back_row[state] = best_previous
            v.append(v_row)
            back.append(back_row)

        best_state = self.states[0]
        best_probability = v[-1][best_state]
        for state in self.states[1:]:
            if v[-1][state] > best_probability:
                best_state = state
                best_probability = v[-1][state]

        path = [best_state]
        for position in range(len(x) - 1, 0, -1):
            path.append(back[position][path[-1]])
        path.reverse()
        return "".join(path), best_probability
    
    #Task 3

    def forward(self, x):
        """Return the log probability of an observation sequence using the forward algorithm."""
        if not x:
            return 0.0

        f = [{state: self.log_pi[state] + self.log_e[state][x[0]] for state in self.states}]

        for symbol in x[1:]:
            f_row = {}
            for state in self.states:
                log_sum_prob = -math.inf
                for previous in self.states:
                    log_sum_prob = log_sum(log_sum_prob, f[-1][previous] + self.log_a[previous][state])
                f_row[state] = log_sum_prob + self.log_e[state][symbol]
            f.append(f_row)

        total_log_probability = -math.inf
        for state in self.states:
            total_log_probability = log_sum(total_log_probability, f[-1][state])

        return total_log_probability

    def backward(self, x):
        """Return the log probability of an observation sequence using the backward algorithm."""
        if not x:
            return 0.0

        b = [{state: 0.0 for state in self.states}]

        for symbol in reversed(x[1:]):
            b_row = {}
            for state in self.states:
                log_sum_prob = -math.inf
                for next_state in self.states:
                    log_sum_prob = log_sum(log_sum_prob, self.log_a[state][next_state] + self.log_e[next_state][symbol] + b[-1][next_state])
                b_row[state] = log_sum_prob
            b.append(b_row)

        b.reverse()

        total_log_probability = -math.inf
        for state in self.states:
            total_log_probability = log_sum(total_log_probability, self.log_pi[state] + self.log_e[state][x[0]] + b[0][state])

        return total_log_probability

    def posterior(self, x):
        """Return the posterior probabilities of states at each position in an observation sequence."""
        if not x:
            return []

        log_forward = self.forward(x)
        log_backward = self.backward(x)
        total_log_probability = log_forward  # or log_backward, they should be equal

        posterior_probs = []
        for position in range(len(x)):
            position_probs = {}
            for state in self.states:
                log_prob = self.log_pi[state] + self.log_e[state][x[position]] + log_backward[position][state]
                position_probs[state] = 2 ** (log_prob - total_log_probability)
            posterior_probs.append(position_probs)

        return posterior_probs

    def posterior_path(self, x):
        """Return the most likely path based on posterior probabilities."""
        posterior_probs = self.posterior(x)
        path = []
        for position_probs in posterior_probs:
            best_state = max(position_probs, key=position_probs.get)
            path.append(best_state)
        return "".join(path)

#print(read_model("weather_model.json").viterbi("ssrssrrrss"))
#print(read_model("weather_model.json").log_joint("ssrssrrrss", "HHHHHLLLHH"))

def run_fasta(fasta_paths, model_path):
    """Compare log-space and plain-probability Viterbi on FASTA records."""
    model = read_model(model_path)
    for fasta_path in fasta_paths:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence = str(record.seq)
            log_path, log_score = model.viterbi(sequence)
            plain_path, plain_probability = model.viterbi_plain(sequence)
            first_difference = next(
                (
                    index + 1
                    for index, (log_state, plain_state) in enumerate(
                        zip(log_path, plain_path)
                    )
                    if log_state != plain_state
                ),
                None,
            )
            bits_per_position = -log_score / len(sequence) if sequence else 0.0
            print(
                f"{fasta_path}:{record.id} "
                f"length={len(sequence)} "
                f"log_score={log_score:.3f} "
                f"bits_per_position={bits_per_position:.3f} "
                f"plain_probability={plain_probability:.6g} "
                f"paths_match={log_path == plain_path} "
                f"first_difference={first_difference}"
            )


def main():
    """Run both Viterbi implementations on one or more FASTA files."""
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare log-space and plain-probability HMM Viterbi."
    )
    parser.add_argument(
        "fasta_files",
        nargs="*",
        type=Path,
        help="FASTA files to decode (default: sequences.fasta)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_dir / "toy_model.json",
        help="JSON HMM model (default: toy_model.json)",
    )
    args = parser.parse_args()
    fasta_paths = args.fasta_files or [default_dir / "sequences.fasta"]
    run_fasta(fasta_paths, args.model)

#if __name__ == "__main__":
#    main()