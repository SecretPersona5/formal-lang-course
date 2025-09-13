# Task 1 and Task 2

from typing import Set, Tuple

import cfpq_data as cd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol


__all__ = [
    "graph_stats",
    "save_two_cycles",
    "graph_to_nfa",
    "save_nfa_dot",
]


def graph_stats(name: str) -> Tuple[int, int, set[str]]:
    path = cd.download(name)
    G = cd.graph_from_csv(path)
    labels = set(nx.get_edge_attributes(G, "label").values())
    return G.number_of_nodes(), G.number_of_edges(), labels


def save_two_cycles(n: int, m: int, a: str, b: str, out_path: str) -> None:
    G = cd.labeled_two_cycles_graph(n, m, labels=(a, b))
    write_dot(G, out_path)


def graph_to_nfa(
    graph: nx.MultiGraph,
    start_states: Set[int],
    final_states: Set[int],
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is None:
            continue
        nfa.add_transition(State(u), Symbol(str(label)), State(v))

    starts = set(graph.nodes) if not start_states else set(start_states)
    finals = set(graph.nodes) if not final_states else set(final_states)

    for s in starts:
        nfa.add_start_state(State(s))
    for f in finals:
        nfa.add_final_state(State(f))

    return nfa


def save_nfa_dot(nfa: NondeterministicFiniteAutomaton, out_path: str) -> None:
    write_dot(nfa.to_networkx(), out_path)
