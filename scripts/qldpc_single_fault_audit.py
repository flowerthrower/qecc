# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Audit single faults in a noisy Stim circuit without decomposing detector hyperedges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim


def audit_single_faults(circuit: stim.Circuit) -> dict[str, object]:
    """Check whether detector outcomes distinguish the declared logical fault effects.

    The input must already contain its noise model, detectors, and observables.
    A pass means that some decoder can correct the declared observables for no
    fault or one noise event from that model. It does not verify the ideal logical
    operation or protect logical information omitted from the observables.
    Conditional channels using ELSE_CORRELATED_ERROR are unsupported because
    Stim's fault explanations do not check branch reachability.

    Args:
        circuit: The complete noisy detector circuit.

    Returns:
        Counts and the first ambiguity witness, with original Stim fault locations.
        Counts omit faults with no detector or observable effect, as Stim does.

    Raises:
        ValueError: The circuit uses conditional noise or has nondeterministic
            detectors or observables.
    """
    if any(operation.name == "ELSE_CORRELATED_ERROR" for operation in circuit.flattened()):
        msg = "Conditional noise using ELSE_CORRELATED_ERROR is unsupported."
        raise ValueError(msg)
    circuit.without_noise().detector_error_model()
    errors = circuit.explain_detector_error_model_errors(reduce_to_one_representative_error=False)
    seen: dict[tuple[int, ...], tuple[tuple[int, ...], stim.CircuitErrorLocation | None]] = {(): ((), None)}
    ambiguous: set[tuple[int, ...]] = set()
    witness = None
    max_detectors = 0
    for error in errors:
        targets = [term.dem_target for term in error.dem_error_terms]
        detectors = tuple(target.val for target in targets if target.is_relative_detector_id())
        observables = tuple(target.val for target in targets if target.is_logical_observable_id())
        location = error.circuit_error_locations[0]
        max_detectors = max(max_detectors, len(detectors))
        if detectors not in seen:
            seen[detectors] = (observables, location)
            continue
        previous_observables, previous_location = seen[detectors]
        if previous_observables != observables:
            ambiguous.add(detectors)
            if witness is None:
                witness = {
                    "detectors": list(detectors),
                    "first": {
                        "observables": list(previous_observables),
                        "location": str(previous_location) if previous_location is not None else None,
                    },
                    "second": {"observables": list(observables), "location": str(location)},
                }
    return {
        "scope": "Declared observables; no fault or one noise event in the supplied circuit.",
        "one_fault_correctable": not ambiguous,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "nontrivial_signatures": len(errors),
        "nontrivial_fault_alternatives": sum(len(error.circuit_error_locations) for error in errors),
        "max_detectors_per_fault": max_detectors,
        "ambiguous_detector_signatures": len(ambiguous),
        "witness": witness,
    }


def main() -> None:
    """Print a JSON audit for a fully prepared noisy Stim circuit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path, help="Noisy .stim circuit with detectors and observables.")
    args = parser.parse_args()
    result = audit_single_faults(stim.Circuit.from_file(args.circuit))
    print(json.dumps({"circuit": str(args.circuit), **result}, indent=2))


if __name__ == "__main__":
    main()
