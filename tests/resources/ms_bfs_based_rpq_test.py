# tests for task4
import pytest
import networkx as nx
from project.task4 import ms_bfs_based_rpq


def make_graph(edges):
    g = nx.MultiDiGraph()
    for u, lbl, v in edges:
        g.add_edge(u, v, label=lbl)
    return g


def test_loop_a_star_epsilon_ok():
    g = make_graph([(0, "a", 0)])
    res = ms_bfs_based_rpq("a*", g, {0}, {0})
    assert res == {(0, 0)}


def test_loop_a_matches():
    g = make_graph([(0, "a", 0)])
    res = ms_bfs_based_rpq("a", g, {0}, {0})
    assert res == {(0, 0)}


def test_simple_edge_a():
    g = make_graph([(0, "a", 1)])
    res = ms_bfs_based_rpq("a", g, {0}, {1})
    assert res == {(0, 1)}


def test_concat_ab():
    g = make_graph([(0, "a", 1), (1, "b", 2)])
    res = ms_bfs_based_rpq("a b", g, {0}, {2})
    assert res == {(0, 2)}


def test_union_a_or_b():
    g = make_graph([(0, "a", 1), (0, "b", 2)])
    res = ms_bfs_based_rpq("(a | b)", g, {0}, {1, 2})
    assert res == {(0, 1), (0, 2)}


def test_kleene_chain_a_star_transitive_closure():
    g = make_graph([(0, "a", 1), (1, "a", 2)])
    res = ms_bfs_based_rpq("a*", g, {0, 1, 2}, {0, 1, 2})
    expected = {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    assert res == expected


def test_no_match_if_label_absent():
    g = make_graph([(0, "a", 1), (1, "b", 2)])
    res = ms_bfs_based_rpq("c", g, {0, 1, 2}, {0, 1, 2})
    assert res == set()


def test_kleene_on_tail_abstar_cstar():
    g = make_graph(
        [
            (0, "a", 1),
            (1, "b", 2),
            (2, "b", 2),
            (2, "c", 3),
        ]
    )
    res = ms_bfs_based_rpq("a b* c*", g, {0}, {1, 2, 3})
    assert {(0, 1), (0, 2), (0, 3)}.issubset(res)


@pytest.mark.parametrize(
    "regex, edges, starts, finals, expected",
    [
        (
            "a",
            [(0, "a", 1), (1, "a", 2), (2, "b", 3)],
            {0, 1},
            {1, 2},
            {(0, 1), (1, 2)},
        ),
        (
            "(a | c) b",
            [(0, "a", 1), (0, "c", 2), (1, "b", 3), (2, "b", 4)],
            {0},
            {3, 4},
            {(0, 3), (0, 4)},
        ),
        ("a*", [], {0}, {1}, set()),
    ],
)
def test_parametrized(regex, edges, starts, finals, expected):
    g = make_graph(edges)
    res = ms_bfs_based_rpq(regex, g, set(starts), set(finals))
    assert res == expected
