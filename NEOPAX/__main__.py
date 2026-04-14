"""
NEOPAX module entry point: allows 'python -m NEOPAX <config.toml>'
"""
import sys
from .main import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m NEOPAX <config.toml>")
        sys.exit(1)
    main(sys.argv[1])