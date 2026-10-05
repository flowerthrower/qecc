# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the research script's single-fault signature audit."""

from __future__ import annotations

import pytest
import stim

from scripts.qldpc_single_fault_audit import audit_single_faults


def test_unreachable_conditional_fault_is_rejected() -> None:
    """Stim explains the unreachable X branch as a possible logical fault."""
    circuit = stim.Circuit("""
        R 0
        E(1) Z0
        ELSE_CORRELATED_ERROR(0.01) X0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    with pytest.raises(ValueError, match="ELSE_CORRELATED_ERROR"):
        audit_single_faults(circuit)


def test_random_detector_cannot_certify_fault_correction() -> None:
    """A random detector cannot distinguish a logical fault from no fault."""
    circuit = stim.Circuit("""
        R 0 1
        H 1
        E(0.01) X0 X1
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        audit_single_faults(circuit)


def test_same_detectors_with_different_observables() -> None:
    """Two individually detectable faults can require different logical corrections."""
    circuit = stim.Circuit("""
        R 0 1
        X_ERROR(0.01) 0 1
        M 0 1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    result = audit_single_faults(circuit)
    assert result["one_fault_correctable"] is False
    assert result["ambiguous_detector_signatures"] == 1
    witness = result["witness"]
    assert isinstance(witness, dict)
    assert witness["detectors"] == [0]
    assert witness["first"]["observables"] == []
    assert witness["second"]["observables"] == [0]
    assert "X_ERROR(0.01) 1" in witness["first"]["location"]
    assert "X_ERROR(0.01) 0" in witness["second"]["location"]


def test_undetected_measurement_fault_conflicts_with_no_fault() -> None:
    """An observable flip with no detector conflicts with the explicit no-fault case."""
    circuit = stim.Circuit("R 0\nM(0.01) 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    result = audit_single_faults(circuit)
    assert result["one_fault_correctable"] is False
    witness = result["witness"]
    assert isinstance(witness, dict)
    assert witness["detectors"] == []
    assert witness["first"] == {"observables": [], "location": None}
    assert witness["second"]["observables"] == [0]
    assert "flipped_measurement" in witness["second"]["location"]


def test_identical_fault_signatures_are_correctable() -> None:
    """X and Y effects coincide here; the harmless Z alternative may be omitted."""
    circuit = stim.Circuit("""
        R 0
        DEPOLARIZE1(0.01) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    result = audit_single_faults(circuit)
    assert result["one_fault_correctable"] is True
    assert result["nontrivial_signatures"] == 1
    assert result["nontrivial_fault_alternatives"] == 2
    assert result["witness"] is None


def test_two_qubit_fault_alternatives_keep_hyperedges() -> None:
    """Bell probes distinguish all 15 Pauli faults at one two-qubit location."""
    circuit = stim.Circuit("""
        R 0 1 2 3
        H 0 1
        CX 0 2 1 3
        DEPOLARIZE2(0.01) 0 1
        CX 0 2 1 3
        H 0 1
        M 0 1 2 3
        DETECTOR rec[-4]
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-4]
    """)
    result = audit_single_faults(circuit)
    assert result["one_fault_correctable"] is True
    assert result["nontrivial_fault_alternatives"] == 15
    assert result["nontrivial_signatures"] == 15
    assert result["max_detectors_per_fault"] == 4
