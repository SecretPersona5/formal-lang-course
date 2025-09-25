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
    q0 = startB[0]

    finalB_raw = set(B.final_states)
    if not all(isinstance(x, int) for x in finalB_raw):
        finalB = {idxB[x] for x in finalB_raw}
    else:
        finalB = finalB_raw

    idxA: Dict[Hashable, int] = {s: i for i, s in enumerate(A.states)}

    def decode(idx: int) -> tuple[int, int]:
        return idx // nB, idx % nB

    answers: Set[Tuple[int, int]] = set()

    for u in start_nodes:
        if u in idxA:
            iu = idxA[u]
        elif State(u) in idxA:
            iu = idxA[State(u)]
        else:
            continue

        start_prod = iu * nB + q0

        reach = csr_matrix(
            ([True], ([0], [start_prod])),
            shape=(1, U.shape[0]),
            dtype=bool,
        )
        prev_nnz = -1
        while reach.nnz != prev_nnz:
            prev_nnz = reach.nnz
            reach = ((reach + (reach @ U)) > 0).astype(bool)

        dense = reach.toarray()[0]
        where_true = np.where(dense)[0]

        for p in where_true:
            iA, iB = decode(p)
            if iB in finalB:
                g_state = A.states[iA]
                v = g_state.value if isinstance(g_state, State) else g_state
                if v in final_nodes:
                    answers.add((u, int(v)))

    return answers
