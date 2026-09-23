"""Bounded legacy table-transpose probe; no transport, root or scan solve.

The default is the existing three-call baseline. ``--mode compare`` also
measures a diagnostic coefficient-axis vmap over the SAME scalar primitive.
No production interpolation, support hook or CLI execution path is changed.

Example (from the repository root)::

    python tests/benchmark_database_table_scatter_cost.py --device cpu --mode compare
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_case(objectives, energies):
    import jax.numpy as jnp
    import numpy as np
    from NEOPAX._database import Monoenergetic

    rho = jnp.asarray([0.12247, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    nu = jnp.asarray([
        3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
        3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0,
    ])
    er = jnp.linspace(-8.0, 1.0, 11)
    base = jnp.arange(7 * 16 * 11, dtype=jnp.float64).reshape((7, 16, 11))
    database = Monoenergetic(
        a_b=jnp.asarray(1.0), rho=rho, nu_log=jnp.log10(nu),
        # Deliberately radius-dependent: stencils cannot share an Er row.
        Er_list=er[None, :] + 0.03 * rho[:, None],
        D11_log=-3.0 + base * 1e-4, D13=0.2 + base * 2e-4,
        D33=0.4 + base * 3e-4,
    )
    queries_nu = jnp.asarray([0.0, 3e-7, 2.4e-2, 100.0])[:energies]
    queries_er = jnp.asarray([0.0, -1e-7, 2.2e-4, 1.0])[:energies]
    bars = np.random.default_rng(20260912).normal(size=(objectives, energies, 3))
    bars[0] = 0.0
    return database, jnp.asarray(0.51), queries_nu, queries_er, jnp.asarray(bars)


def build_kernels():
    import jax
    import jax.numpy as jnp
    from NEOPAX._interpolators import monoenergetic_interpolation_table_bar

    def kernel(database, radius, queries_nu, queries_er, bars, *, coefficient_vmap):
        tables = (database.D11_log, database.D13, database.D33)

        def scatter_one(nu, er, local_bars):
            if coefficient_vmap:
                # The only candidate change: map the unchanged scalar rule
                # over three coefficients, then restore the tuple contract.
                result = jax.vmap(
                    lambda table, bar: monoenergetic_interpolation_table_bar(
                        radius, nu, er, bar, table, database,
                    )
                )(jnp.stack(tables), local_bars)
                return tuple(result[index] for index in range(3))
            return tuple(
                monoenergetic_interpolation_table_bar(
                    radius, nu, er, local_bars[index], table, database,
                )
                for index, table in enumerate(tables)
            )

        def one_row(row_bars):
            per_energy = jax.vmap(scatter_one)(queries_nu, queries_er, row_bars)
            # Match the centre/face primitive's original energy reduction.
            return tuple(jnp.sum(table, axis=0) for table in per_energy)

        return jax.vmap(one_row)(bars)

    return {
        "baseline": lambda *values: kernel(*values, coefficient_vmap=False),
        "coefficient_vmap": lambda *values: kernel(*values, coefficient_vmap=True),
    }


def measure(name, function, inputs, args):
    import jax

    print(json.dumps({"kernel": name, "phase": "lowering"}), flush=True)
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*inputs)
    lowering_s = time.perf_counter() - started
    print(json.dumps({"kernel": name, "phase": "compiling", "lowering_s": lowering_s}), flush=True)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    result = jax.block_until_ready(compiled(*inputs))
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*inputs))
        samples.append(time.perf_counter() - started)
    hlo = compiled.as_text()
    memory = compiled.memory_analysis()
    memory_fields = (
        "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
        "temp_size_in_bytes", "generated_code_size_in_bytes",
        "host_argument_size_in_bytes", "host_output_size_in_bytes", "host_temp_size_in_bytes",
    )
    report = {
        "kernel": name, "lowering_s": lowering_s, "compile_s": compile_s,
        "warm_median_s": statistics.median(samples), "warm_samples_s": samples,
        "optimized_hlo_bytes": len(hlo.encode()),
        "opcode_counts": dict(sorted(Counter(re.findall(r" = .*?\b([a-z][a-z0-9-]*)\(", hlo)).items())),
        "memory_analysis": {key: getattr(memory, key, None) for key in memory_fields},
        "cost_analysis": compiled.cost_analysis(),
    }
    if args.dump_dir:
        directory = Path(args.dump_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.optimized.hlo.txt").write_text(hlo)
        (directory / f"{name}.stablehlo.txt").write_text(str(lowered.compiler_ir()))
        (directory / f"{name}.json").write_text(json.dumps(report, indent=2))
    cost = report["cost_analysis"]
    if isinstance(cost, list):
        cost = cost[0] if cost else {}
    console_report = {
        **report,
        "cost_analysis": {key: cost.get(key) for key in ("flops", "transcendentals", "bytes accessed")},
        "opcode_counts": {key: report["opcode_counts"].get(key, 0) for key in ("fusion", "while", "dynamic-update-slice", "transpose")},
    }
    print(json.dumps(console_report), flush=True)
    return compiled, result


def check_query_branches(baseline, candidate, inputs):
    import jax
    import jax.numpy as jnp
    import numpy as np

    database, _, queries_nu, queries_er, bars = inputs
    branch_cases = (
        ("axis", 0.0), ("small", 0.13),
        ("below_small_mid_tie", np.nextafter(0.25, 0.0)),
        ("small_mid_tie", 0.25), ("mid", 0.51),
        ("below_mid_large_tie", np.nextafter(0.75, 0.0)),
        ("mid_large_tie", 0.75), ("large", 0.88), ("radial_extrapolation", 1.1),
    )
    reports = []
    for label, radius in branch_cases:
        values = (database, jnp.asarray(radius), queries_nu, queries_er, bars)
        expected = jax.block_until_ready(baseline(*values))
        actual = jax.block_until_ready(candidate(*values))
        max_abs = 0.0
        elementwise_equal = True
        for reference, result in zip(expected, actual, strict=True):
            reference, result = np.asarray(reference), np.asarray(result)
            if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(result))):
                raise AssertionError(f"{label}: nonfinite table cotangent")
            np.testing.assert_allclose(result, reference, rtol=2e-12, atol=2e-12)
            np.testing.assert_array_equal(result[0], np.zeros_like(result[0]))
            max_abs = max(max_abs, float(np.max(np.abs(result - reference))))
            elementwise_equal = elementwise_equal and np.array_equal(result, reference)
        report = {"query_branch": label, "radius": radius, "finite": True, "max_abs": max_abs, "elementwise_equal": elementwise_equal}
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--mode", choices=("baseline", "coefficient_vmap", "compare"), default="baseline")
    parser.add_argument("--objectives", type=int, default=10)
    parser.add_argument("--energies", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dump-dir")
    args = parser.parse_args()
    if not (1 <= args.objectives <= 10 and 1 <= args.energies <= 4 and 1 <= args.repeats <= 100):
        parser.error("Bounded probe requires 1..10 objectives, 1..4 energies, and 1..100 repeats.")
    # Set before importing the package, whose module constants initialize JAX.
    # This workspace's GPU environment is CUDA-only. The generic "gpu"
    # platform alias also attempts ROCm when explicitly requested in JAX 0.5.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    inputs = prepare_case(args.objectives, args.energies)
    kernels = build_kernels()
    print(json.dumps({
        "fixture": "synthetic runtime-shaped Monoenergetic tables; unchanged scalar table transpose; no transport/root/scan",
        "jax": jax.__version__, "device": str(jax.devices()[0]),
        "table_shape": list(inputs[0].D11_log.shape), "objectives": args.objectives,
        "energies": args.energies, "radius_queries": 1, "mode": args.mode,
    }), flush=True)
    modes = ("baseline", "coefficient_vmap") if args.mode == "compare" else (args.mode,)
    compiled = {}
    for mode in modes:
        compiled[mode], result = measure(mode, kernels[mode], inputs, args)
        if not all(np.all(np.isfinite(value)) for value in result):
            raise AssertionError(f"{mode}: nonfinite default query result")
    if args.mode == "compare":
        reports = check_query_branches(compiled["baseline"], compiled["coefficient_vmap"], inputs)
        if args.dump_dir:
            (Path(args.dump_dir) / "parity.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
