# Tests for task 2

import pytest
import networkx as nx

from project.init_graph import graph_to_nfa

pytest.importorskip("pyformlang")


def test_graph_to_nfa_defaults_all_states_are_start_and_final():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, label="a")
    G.add_edge(1, 2, label="b")

    nfa = graph_to_nfa(G, start_states=set(), final_states=set())

    starts = {s.value for s in nfa.start_states}
    finals = {f.value for f in nfa.final_states}

    assert starts == set(G.nodes)
    assert finals == set(G.nodes)

    assert nfa.accepts([])


def test_graph_to_nfa_acceptance_on_simple_path_and_loop():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, label="a")
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 2, label="b")

    nfa = graph_to_nfa(G, start_states={0}, final_states={2})

    assert nfa.accepts(["a", "b"])
    assert nfa.accepts(["a", "b", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b"])


def test_graph_to_nfa_is_nondeterministic_with_parallel_edges():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, label="x")
    G.add_edge(0, 2, label="x")

    nfa = graph_to_nfa(G, start_states={0}, final_states={1, 2})
    assert nfa.is_deterministic() is False
    assert nfa.accepts(["x"])


def test_graph_to_nfa_ignores_unlabeled_edges():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, label="a")
    G.add_edge(1, 2, label="b")
    G.add_edge(0, 2)

    nfa = graph_to_nfa(G, start_states={0}, final_states={2})

    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["b"])


def test_graph_to_nfa_on_two_cycles_dataset_accepts_single_step():
    cd = pytest.importorskip("cfpq_data")

    G = cd.labeled_two_cycles_graph(3, 4, labels=("a", "b"))
    nfa = graph_to_nfa(G, start_states=set(), final_states=set())

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
