# Tests for task1

from pathlib import Path
import pytest

from project.init_graph import graph_stats, save_two_cycles


def test_graph_stats_bzip_truth():
    nodes, edges, labels = graph_stats("bzip")
    assert nodes == 632
    assert edges == 556
    assert labels == {"a", "d"}
    assert isinstance(labels, set) and all(isinstance(s, str) for s in labels)

pydot = pytest.importorskip("pydot")


def test_save_two_cycles_creates_GR_dot():
    n, m = 10, 15
    out = Path(__file__).parent / "resources" / "GR.dot"
    out.parent.mkdir(parents=True, exist_ok=True)

    save_two_cycles(n, m, "a", "b", str(out))

    assert out.exists() and out.stat().st_size > 0

    graphs = pydot.graph_from_dot_file(str(out))
    g = graphs[0]

    assert len(g.get_nodes()) == n + m + 1
    assert len(g.get_edges()) == n + m + 2

    edge_labels = {e.get_attributes().get("label") for e in g.get_edges()}
    assert {"a", "b"}.issubset(edge_labels)
