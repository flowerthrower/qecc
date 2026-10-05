# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for equivalence-checking decision procedures."""

from __future__ import annotations

from collections import Counter
from functools import reduce
from typing import TYPE_CHECKING, overload

import networkx as nx
import numpy as np
import z3

from ..codes.core.css_code import CSSCode
from ..codes.core.pauli import PauliTableau
from ..codes.core.stabilizer_code import StabilizerCode
from ..mod2 import row_basis

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def _preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-qubits invariant."""
    return c1.n == c2.n


def _preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-logical-qubits invariant."""
    return c1.k == c2.k


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def _elementwise_map(normal_bool: npt.NDArray[np.integer], variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    """Constrain Boolean variables to equal a binary vector."""
    return z3.And([
        variable if bit == 1 else z3.Not(variable) for bit, variable in zip(normal_bool, variables, strict=True)
    ])


def _exactly_one(variables: Iterable[z3.BoolRef]) -> z3.BoolRef:
    """Constrain exactly one of the given Boolean variables to hold."""
    return z3.PbEq([(variable, 1) for variable in variables], 1)


def _encode_row_operations(
    solver: z3.Solver,
    auxiliary_matrix: Sequence[z3.BoolRef],
    target_matrix: npt.NDArray[np.integer],
    *,
    variable_prefix: str,
) -> None:
    """Constrain an auxiliary matrix to lie in the target matrix's row space."""
    rows, columns = target_matrix.shape
    coefficients = [z3.Bool(f"{variable_prefix}_{row}_{source}") for row in range(rows) for source in range(rows)]

    for row in range(rows):
        for column in range(columns):
            contributions = (
                coefficients[row * rows + source] for source in range(rows) if target_matrix[source, column] == 1
            )
            solver.add(auxiliary_matrix[row * columns + column] == reduce(z3.Xor, contributions, z3.BoolVal(False)))


def _colored_graph_isomorphism(
    graph1: nx.Graph, graph2: nx.Graph, *, edge_colors: bool = False
) -> dict[int, int] | None:
    """Find an isomorphism that preserves the ``color`` attribute of nodes and, optionally, edges.

    The graphs have to be either both simple graphs or both multigraphs. In multigraphs,
    the colors of the parallel edges between two nodes have to be preserved as a set.

    The node colors are first refined with the Weisfeiler-Lehman procedure. Differing refinements
    refute an isomorphism, while matching ones prune the search of the VF2 matcher, which otherwise
    backtracks heavily on the highly symmetric incidence graphs of codes.
    Both graphs are annotated with the refinement as ``refined_color`` node attribute.

    Returns:
        The node mapping from ``graph1`` to ``graph2``, or ``None`` if the graphs are not isomorphic.
    """
    for graph in (graph1, graph2):
        # the Weisfeiler-Lehman procedure of networkx does not support multigraphs
        hashes = nx.weisfeiler_lehman_subgraph_hashes(
            _merge_parallel_edges(graph) if isinstance(graph, nx.MultiGraph) else graph,
            node_attr="color",
            edge_attr="color" if edge_colors else None,
        )
        nx.set_node_attributes(graph, {node: node_hashes[-1] for node, node_hashes in hashes.items()}, "refined_color")

    if Counter(nx.get_node_attributes(graph1, "refined_color").values()) != Counter(
        nx.get_node_attributes(graph2, "refined_color").values()
    ):
        return None

    isomorphism = nx.algorithms.isomorphism
    categorical_edge_match = (
        isomorphism.categorical_multiedge_match if graph1.is_multigraph() else isomorphism.categorical_edge_match
    )
    matcher = isomorphism.GraphMatcher(
        graph1,
        graph2,
        node_match=isomorphism.categorical_node_match(["color", "refined_color"], [None, None]),
        edge_match=categorical_edge_match("color", None) if edge_colors else None,
    )
    return matcher.mapping if matcher.is_isomorphic() else None


def _merge_parallel_edges(graph: nx.MultiGraph) -> nx.Graph:
    """Merge parallel edges into single edges that are colored by the sorted colors of the merged edges."""
    merged = nx.Graph()
    merged.add_nodes_from(graph.nodes(data=True))
    for node, neighbors in graph.adjacency():
        for neighbor, parallel_edges in neighbors.items():
            colors = sorted(str(data.get("color")) for data in parallel_edges.values())
            merged.add_edge(node, neighbor, color=tuple(colors))
    return merged


@overload
def _reduce_stabilizer_generators(code: CSSCode) -> CSSCode: ...


@overload
def _reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode: ...


def _reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode:
    """Return an equivalent code with a minimal independent generator set."""
    if isinstance(code, CSSCode):
        return CSSCode(
            Hx=row_basis(code.Hx).astype(np.int8),
            Hz=row_basis(code.Hz).astype(np.int8),
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
            Lx=code.Lx,
            Lz=code.Lz,
        )

    reduced_symplectic = row_basis(code.symplectic).astype(np.int8)
    return StabilizerCode(
        generators=PauliTableau.from_matrix(reduced_symplectic),
        distance=code.distance,
        z_logicals=code.z_logicals,
        x_logicals=code.x_logicals,
    )
