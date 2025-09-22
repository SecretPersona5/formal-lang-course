# Task 3

from typing import Dict, Iterable, Set, Hashable, List, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix, kron
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton as NFA,
    DeterministicFiniteAutomaton as DFA,
    State,
    Symbol,
)
from networkx import MultiDiGraph
from project.init_graph import graph_to_nfa
from project.task2 import regex_to_dfa


def _sym_key(s: Hashable) -> Hashable:
    return s.value if isinstance(s, Symbol) else s


class AdjacencyMatrixFA:
    def __init__(self, automaton: Union[NFA, DFA]):
        states = list(automaton.states)
        self.states: List[Hashable] = states
        idx: Dict[Hashable, int] = {s: i for i, s in enumerate(states)}
        n = len(states)
        self.alphabet: Set[Hashable] = {
            _sym_key(a) for a in getattr(automaton, "symbols", set())
        }
        mats: Dict[Hashable, dok_matrix] = {
            a: dok_matrix((n, n), dtype=bool) for a in self.alphabet
        }
        for u, trans in automaton.to_dict().items():
            for sym, vs in trans.items():
                if sym is None:
                    continue
                a = _sym_key(sym)
                targets = vs if isinstance(vs, (set, list, tuple)) else [vs]
                for v in targets:
                    mats.setdefault(a, dok_matrix((n, n), dtype=bool))
                    mats[a][idx[u], idx[v]] = True
        self.matrices: Dict[Hashable, csc_matrix] = {
            a: M.tocsc() for a, M in mats.items()
        }
        self.start_states: Set[int] = {idx[s] for s in automaton.start_states}
        self.final_states: Set[int] = {idx[s] for s in automaton.final_states}

    def _row_mask(self, ids: Iterable[int]) -> csr_matrix:
        n = len(self.states)
        cols = list(ids)
        if not cols:
            return csr_matrix((1, n), dtype=bool)
        data = np.ones(len(cols), dtype=bool)
        rows = [0] * len(cols)
        return csr_matrix((data, (rows, cols)), shape=(1, n), dtype=bool)

    def _mask_array(self, ids: Iterable[int]) -> np.ndarray:
        n = len(self.states)
        m = np.zeros(n, dtype=bool)
        for i in ids:
            m[i] = True
        return m

    def accepts(self, word: Iterable[Hashable]) -> bool:
        cur = self._row_mask(self.start_states)
        for sym in word:
            a = _sym_key(sym)
            M = self.matrices.get(a)
            if M is None:
                return False
            cur = (cur @ M).astype(bool)
            if cur.nnz == 0:
                return False
        finals = self._mask_array(self.final_states)
        return bool(np.any(cur.toarray()[0] & finals))

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True
        if not self.matrices:
            return len(self.start_states & self.final_states) == 0
        U = None
        for M in self.matrices.values():
            Mi = M.tocsr()
            U = Mi if U is None else ((U + Mi).astype(bool))
        reach = self._row_mask(self.start_states)
        prev = -1
        while reach.nnz != prev:
            prev = reach.nnz
            reach = ((reach + (reach @ U)) > 0).astype(bool)
        finals = self._mask_array(self.final_states)
        return not bool(np.any(reach.toarray()[0] & finals))


def intersect_automata(
    A: "AdjacencyMatrixFA", B: "AdjacencyMatrixFA"
) -> "AdjacencyMatrixFA":
    nA, nB = len(A.states), len(B.states)
    alphabet = set(A.alphabet) | set(B.alphabet)
    matrices: Dict[Hashable, csc_matrix] = {}
    for a in alphabet:
        MA = A.matrices.get(a, csc_matrix((nA, nA), dtype=bool))
        MB = B.matrices.get(a, csc_matrix((nB, nB), dtype=bool))
        matrices[a] = kron(MA, MB, format="csc")
    start_pairs = {iA * nB + iB for iA in A.start_states for iB in B.start_states}
    final_pairs = {iA * nB + iB for iA in A.final_states for iB in B.final_states}
    prod = AdjacencyMatrixFA.__new__(AdjacencyMatrixFA)
    prod.states = [(sa, sb) for sa in A.states for sb in B.states]
    prod.alphabet = set(alphabet)
    prod.start_states = start_pairs
    prod.final_states = final_pairs
    prod.matrices = matrices
    return prod


def tensor_based_rpq(
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
        U = Mi if U is None else ((U + Mi).astype(bool))
    _, nB = len(A.states), len(B.states)
    startB = list(B.start_states)
    if not startB:
        return set()
    q0 = startB[0]
    finalB = set(B.final_states)
    idxA = {s: i for i, s in enumerate(A.states)}

    def decode(p: int) -> Tuple[int, int]:
        return p // nB, p % nB

    answers: Set[Tuple[int, int]] = set()
    for u in start_nodes:
        if u in idxA:
            iu = idxA[u]
        elif State(u) in idxA:
            iu = idxA[State(u)]
        else:
            continue
        start_p = iu * nB + q0
        reach = csr_matrix(
            ([True], ([0], [start_p])), shape=(1, U.shape[0]), dtype=bool
        )
        prev = -1
        while reach.nnz != prev:
            prev = reach.nnz
            reach = ((reach + (reach @ U)) > 0).astype(bool)
        dense = reach.toarray()[0].astype(bool)
        where = np.where(dense)[0]
        for p in where:
            iA, iB = decode(p)
            if iB in finalB:
                g_state = A.states[iA]
                v = g_state.value if isinstance(g_state, State) else g_state
                if v in final_nodes:
                    answers.add((u, v))
    return answers
