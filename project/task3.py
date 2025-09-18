# Task 3

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Hashable, List, Tuple

import numpy as np
import networkx as nx
from scipy.sparse import dok_matrix, csr_matrix, spmatrix, kron

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)

from project.init_graph import graph_to_nfa
from project.task2 import regex_to_dfa


@dataclass
class AdjacencyMatrixFA:
    states: List[Hashable]
    alphabet: Set[Hashable]
    start_states: Set[int]
    final_states: Set[int]
    matrices: Dict[Hashable, csr_matrix]

    @classmethod
    def from_dfa(cls, dfa: DeterministicFiniteAutomaton) -> "AdjacencyMatrixFA":
        return cls._from_fa_core(dfa)

    @classmethod
    def from_nfa(cls, nfa: NondeterministicFiniteAutomaton) -> "AdjacencyMatrixFA":
        return cls._from_fa_core(nfa)

    @classmethod
    def _from_fa_core(cls, fa) -> "AdjacencyMatrixFA":
        states = [s.value if isinstance(s, State) else s for s in fa.states]
        index = {s: i for i, s in enumerate(states)}
        n = len(states)

        alphabet: Set[Hashable] = set()
        for s_from, sym in fa.to_dict().keys():
            a = sym.value if isinstance(sym, Symbol) else sym
            alphabet.add(a)

        mats: Dict[Hashable, dok_matrix] = {
            a: dok_matrix((n, n), dtype=bool) for a in alphabet
        }

        for (s_from, sym), tos in fa.to_dict().items():
            a = sym.value if isinstance(sym, Symbol) else sym
            i = index[s_from.value if isinstance(s_from, State) else s_from]
            for s_to in tos:
                j = index[s_to.value if isinstance(s_to, State) else s_to]
                mats[a][i, j] = True

        start_states = {
            index[s.value if isinstance(s, State) else s] for s in fa.start_states
        }
        final_states = {
            index[s.value if isinstance(s, State) else s] for s in fa.final_states
        }

        csr_mats = {a: m.tocsr() for a, m in mats.items()}
        return cls(
            states=states,
            alphabet=alphabet,
            start_states=start_states,
            final_states=final_states,
            matrices=csr_mats,
        )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        n = len(self.states)
        if n == 0 or not self.start_states or not self.final_states:
            return False

        front = dok_matrix((1, n), dtype=bool)
        for i in self.start_states:
            front[0, i] = True
        front = front.tocsr()

        for raw_sym in word:
            a = raw_sym.value if isinstance(raw_sym, Symbol) else raw_sym
            M = self.matrices.get(a)
            if M is None:
                return False
            front = (front @ M).astype(bool)
            if front.nnz == 0:
                return False

        cols = np.array(list(self.final_states))
        return front[:, cols].nnz > 0

    def is_empty(self) -> bool:
        n = len(self.states)
        if n == 0 or not self.start_states or not self.final_states:
            return True

        R = dok_matrix((1, n), dtype=bool)  # достигнутые
        for i in self.start_states:
            R[0, i] = True
        R = R.tocsr()
        frontier = R.copy()

        if self._has_final(frontier):
            return False

        while True:
            next_frontier = csr_matrix((1, n), dtype=bool)
            for M in self.matrices.values():
                step = (frontier @ M).astype(bool)
                next_frontier = ((next_frontier + step) > 0).astype(bool)

            new = ((next_frontier.astype(int) - R.astype(int)) > 0).astype(bool)
            if new.nnz == 0:
                break
            R = ((R + new) > 0).astype(bool)
            frontier = new

            if self._has_final(frontier):
                return False

        return True

    def _has_final(self, row_mat: spmatrix) -> bool:
        if not self.final_states:
            return False
        cols = np.array(list(self.final_states))
        return row_mat[:, cols].nnz > 0


def intersect_automata(
    a1: AdjacencyMatrixFA, a2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    common = a1.alphabet & a2.alphabet
    n1, n2 = len(a1.states), len(a2.states)

    def idx(i: int, j: int) -> int:
        return i * n2 + j

    start = {idx(i, j) for i in a1.start_states for j in a2.start_states}
    final = {idx(i, j) for i in a1.final_states for j in a2.final_states}

    matrices: Dict[Hashable, csr_matrix] = {}
    for a in common:
        M = kron(a1.matrices[a], a2.matrices[a], format="csr")
        matrices[a] = M.astype(bool)

    states = [(i, j) for i in range(n1) for j in range(n2)]
    return AdjacencyMatrixFA(
        states=states,
        alphabet=common,
        start_states=start,
        final_states=final,
        matrices=matrices,
    )


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> Set[Tuple[int, int]]:
    dfa: DeterministicFiniteAutomaton = regex_to_dfa(regex)
    am_regex = AdjacencyMatrixFA.from_dfa(dfa)

    nfa_graph = graph_to_nfa(graph, start_states=start_nodes, final_states=final_nodes)
    am_graph = AdjacencyMatrixFA.from_nfa(nfa_graph)

    am_inter = intersect_automata(am_graph, am_regex)

    n_graph = len(am_graph.states)
    n_regex = len(am_regex.states)
    n_inter = len(am_inter.states)

    final_cols = set()
    for i in range(n_graph):
        for qf in am_regex.final_states:
            final_cols.add(i * n_regex + qf)
    final_cols = np.array(list(final_cols), dtype=int)

    ans: Set[Tuple[int, int]] = set()

    for u in start_nodes:
        row = dok_matrix((1, n_inter), dtype=bool)
        for q0 in am_regex.start_states:
            col = u * n_regex + q0
            row[0, col] = True
        row = row.tocsr()

        R = row.copy()
        frontier = row.copy()

        if frontier[:, final_cols].nnz > 0:
            ans.add((u, u))

        while True:
            next_frontier = csr_matrix((1, n_inter), dtype=bool)
            for M in am_inter.matrices.values():
                step = (frontier @ M).astype(bool)
                next_frontier = ((next_frontier + step) > 0).astype(bool)

            new = ((next_frontier.astype(int) - R.astype(int)) > 0).astype(bool)
            if new.nnz == 0:
                break

            R = ((R + new) > 0).astype(bool)
            frontier = new

            hits = new[:, final_cols]
            if hits.nnz > 0:
                _, cols = hits.nonzero()
                used_cols = final_cols[cols]
                vs = (used_cols // n_regex).tolist()
                for v in vs:
                    ans.add(
                        (
                            u,
                            int(am_graph.states[v])
                            if isinstance(am_graph.states[v], int)
                            else v,
                        )
                    )

    return {(int(u), int(v)) for (u, v) in ans}
