"""Main command line interface to SPECTRAX."""
import sys
from ._plot import plot
from ._simulation import simulation
from ._initialization import load_parameters

def main(cl_args=sys.argv[1:]):
    """Run the main SPECTRAX code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting simulation.

    """
    if len(cl_args) == 0:
        print("Using standard input parameters instead of an input TOML file.")
        output = simulation()
    else:
        input_parameters, solver_parameters = load_parameters(cl_args[0])
        output = simulation(input_parameters, **solver_parameters)
    plot(output)

if __name__ == "__main__":
    main(sys.argv[1:])