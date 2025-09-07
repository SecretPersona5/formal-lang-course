# Task 1

from typing import Tuple

import cfpq_data as cd
import networkx as nx


def graph_stats(name:str) -> Tuple[int, int, set[str]]:
    path = cd.download(name)
    G = cd.graph_from_csv(path)
    labels = set(nx.get_edge_attributes(G, "label").values())
    return G.number_of_nodes(), G.number_of_edges(), labels

def save_two_cycles(n: int, m: int, a:str, b: str, out_path: str) -> None:
    G = cd.labeled_two_cycles_graph(n, m, labels=(a, b))
    from networkx.drawing.nx_pydot import write_dot
    write_dot(G, out_path)
