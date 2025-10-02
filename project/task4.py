# Task4

from typing import Dict, Set, Hashable, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton as NFA,
    DeterministicFiniteAutomaton as DFA,
    State,
)

from project.init_graph import graph_to_nfa
from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> Set[Tuple[int, int]]:
    g_nfa: NFA = graph_to_nfa(graph, start_nodes, final_nodes)
    r_dfa: DFA = regex_to_dfa(regex)

    A = AdjacencyMatrixFA(g_nfa)
    B = AdjacencyMatrixFA(r_dfa)
    P = intersect_automata(A, B)

    U = None
    for M in P.matrices.values():
        Mi = M.tocsr()
        U = Mi if U is None else (U + Mi)
    if U is None:
        return set()
    U = U.astype(bool)

    nB = len(B.states)

    idxB: Dict[Hashable, int] = {s: i for i, s in enumerate(B.states)}
    startB: List[int] = list(B.start_states)
    if not startB:
        return set()
    if not all(isinstance(x, int) for x in startB):
        startB = [idxB[x] for x in startB]

    finalB_raw = set(B.final_states)
    if not all(isinstance(x, int) for x in finalB_raw):
        finalB = {idxB[x] for x in finalB_raw}
    else:
        finalB = finalB_raw

    idxA: Dict[Hashable, int] = {s: i for i, s in enumerate(A.states)}

    def normalize_graph_state(node: Hashable) -> Hashable:
        return node.value if isinstance(node, State) else node

    def get_state_index(node: Hashable) -> int | None:
        if node in idxA:
            return idxA[node]
        if isinstance(node, State):
            return idxA.get(node)
        state_node = State(node)
        return idxA.get(state_node)

    def decode(idx: int) -> tuple[int, int]:
        return idx // nB, idx % nB

    answers: Set[Tuple[int, int]] = set()

    normalized_finals = {normalize_graph_state(v) for v in final_nodes}

    start_rows: List[Hashable] = []
    data: List[bool] = []
    rows: List[int] = []
    cols: List[int] = []

    for u in start_nodes:
        iu = get_state_index(u)
        if iu is None:
            continue
        row_idx = len(start_rows)
        start_rows.append(normalize_graph_state(u))
        for q0 in startB:
            rows.append(row_idx)
            cols.append(iu * nB + q0)
            data.append(True)

        if not rows:
            return set()

        reach = csr_matrix(
            (data, (rows, cols)), shape=(len(start_rows), U.shape[0]), dtype=bool
        )

        prev_nnz = -1
        while reach.nnz != prev_nnz:
            prev_nnz = reach.nnz
            reach = ((reach + (reach @ U)) > 0).astype(bool)

        dense = reach.toarray()

        for row_idx, u in enumerate(start_rows):
            where_true = np.where(dense[row_idx])[0]
            for p in where_true:
                iA, iB = decode(p)
                if iB not in finalB:
                    continue
                g_state = A.states[iA]
                v = normalize_graph_state(g_state)
                if v in normalized_finals:
                    answers.add((int(u), int(v)))

    return answers
