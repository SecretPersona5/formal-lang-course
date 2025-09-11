from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    enfa = Regex(regex).to_epsilon_nfa()
    dfa_min = enfa.minimize()
    return dfa_min
