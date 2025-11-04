# project/task6.py
from collections import defaultdict
from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon
import networkx as nx


__all__ = ["cfg_to_weak_normal_form", "hellings_based_cfpq"]


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nullable = cfg.get_nullable_symbols()
    eps_rules = {Production(v, [Epsilon()]) for v in nullable}
    cfg_nf = cfg.to_normal_form()
    cfg_nf = CFG(
        variables=cfg_nf.variables,
        terminals=cfg_nf.terminals,
        start_symbol=cfg_nf.start_symbol,
        productions=cfg_nf.productions | eps_rules,
    )
    return cfg_nf.remove_useless_symbols()


def _init_edges(cfg: CFG, graph: nx.MultiDiGraph | nx.DiGraph):
    term_map = defaultdict(set)
    for p in cfg.productions:
        if len(p.body) == 1 and isinstance(p.body[0], Terminal):
            term_map[p.body[0].value].add(p.head)

    edges = set()
    for u, v, label in graph.edges(data="label"):
        for A in term_map.get(label, ()):
            edges.add((u, A, v))

    for v in graph.nodes:
        for A in cfg.get_nullable_symbols():
            edges.add((v, A, v))
    return edges


def _closure(cfg: CFG, edges: set[tuple[int, Variable, int]]):
    bin_rules = [p for p in cfg.productions if len(p.body) == 2]
    added = True
    while added:
        added = False
        new_edges = set()
        for u1, B, v1 in edges:
            for u2, C, v2 in edges:
                if v1 == u2:
                    for p in bin_rules:
                        if p.body[0] == B and p.body[1] == C:
                            e = (u1, p.head, v2)
                            if e not in edges:
                                new_edges.add(e)
        if new_edges:
            edges |= new_edges
            added = True
    return edges


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.MultiDiGraph | nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    cfg = cfg_to_weak_normal_form(cfg)
    edges = _init_edges(cfg, graph)
    edges = _closure(cfg, edges)

    start_nodes = start_nodes or set(graph.nodes)
    final_nodes = final_nodes or set(graph.nodes)
    start = cfg.start_symbol

    result = set()
    for u, A, v in edges:
        if A == start and u in start_nodes and v in final_nodes:
            result.add((u, v))
    return result
