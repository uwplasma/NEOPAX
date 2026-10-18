"""Compare recovered CLI arguments with defaults using only the standard library.

Only the script's parser and its literal default constants are evaluated.
NEOPAX, JAX, VMEX and NTX are not imported; no physics is evaluated.
"""
import argparse
import ast
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    parser_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "parser")
    defaults = {
        keyword.value.id
        for node in ast.walk(parser_node) if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Name)
    }
    namespace = {"argparse": argparse, "Path": Path, "ROOT": root}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in defaults:
                namespace[name] = eval(compile(ast.Expression(node.value), str(source), "eval"), namespace)
    parser_helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_positive_step_count"
    ]
    exec(compile(ast.Module(body=[*parser_helpers, parser_node], type_ignores=[]), str(source), "exec"), namespace)
    parser = namespace["parser"]()
    bare = vars(parser.parse_args([]))
    previous = vars(parser.parse_args([
        "--no-max-er", "--no-root-objectives", "--transition-location-objective",
        "--bootstrap-penalty", "--net-power", "--max-nfev", "30",
        "--out-dir", "outputs/geometry_only_net_power_root_bootstrap_replay_fix",
    ]))
    differences = {name: (bare[name], previous[name]) for name in bare if bare[name] != previous[name]}
    assert set(differences) == {"out_dir"}, differences
    print("Bare command and recovered command: all optimization parser settings match.")
    for name in ("max_er", "root_objectives", "transition_location_objective", "bootstrap_penalty", "net_power", "max_nfev"):
        print(f"{name}: {bare[name]}")
    print("database grid:", tuple(bare[name] for name in ("database_n_theta", "database_n_phi", "database_n_xi")))
    print("Only parser difference: out_dir (output destination).")
    print("Environment differences are outside argparse: reverse diagnostics, snapshots, XLA_FLAGS, PYTHONPATH.")


if __name__ == "__main__":
    main()
