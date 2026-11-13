"""Check TOML parity using only the standard library; no physics imports."""
from copy import deepcopy
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
examples = root / "examples"

def read(path):
    with path.open("rb") as stream:
        return tomllib.load(stream)

source = read(examples / "Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_match_THe.toml")
path = examples / "Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_ntss_match_generic.toml"
actual = read(path)
expected = deepcopy(source)
expected["neoclassical"]["interpolation_mode"] = "generic"
expected["transport_solver"]["temperature_convection_temperature_mode"] = "upwind"
expected["ambipolarity"]["er_ambipolar_output_dir"] = "./outputs/wHe_ntss_first_four_generic_ambipolarity"
expected["transport_output"]["transport_output_dir"] = "./outputs/wHe_ntss_first_four_generic"
assert actual == expected, "Unexpected difference outside interpolation, upwind choice or output paths"
assert actual["neoclassical"]["flux_model"] == "ntx_database"
assert not any(key.startswith("ntx_scan_") for key in actual["neoclassical"])
assert actual["equations"]["match_THe"] is True
assert actual["equations"]["match_workHe"] is True
assert actual["sources"]["parameters"]["bremsstrahlung_radiation"]["coefficient_mode"] == "ntssfusion"
assert actual["transport_solver"]["Er_diffusion_mode"] == "ntssfusion"
assert actual["geometry"]["n_radial"] == 51
assert actual["energy_grid"]["n_x"] == 4
assert actual["transport_solver"]["t_final"] == 20.0
for section, keys in {
    "geometry": ("vmec_file", "boozer_file"),
    "neoclassical": ("neoclassical_file",),
    "ambipolarity": ("er_ambipolar_reference_er_file",),
    "transport_output": ("transport_initial_reference_file", "transport_final_reference_file"),
}.items():
    for key in keys:
        assert (root / actual[section][key]).is_file(), (section, key)
assert all(line == line.rstrip() for line in path.read_text().splitlines()), "Trailing whitespace"
print("PASS: TOML parses; same saved database, generic interpolation, no runtime scan configuration.")
print("PASS: first four NTSS options, upwind convection, 51 radii, four energy points, t_final=20.")
print("PASS: all remaining settings match the saved-table case; inputs exist; output paths are separate.")
print("No NEOPAX/JAX imports or VMEC, NTX, transport, AD or FD solves were run.")
