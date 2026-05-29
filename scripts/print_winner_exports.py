#!/usr/bin/env python3
"""
Read winners.json[arch] and print shell-quoted export statements.
Used by slurm_scripts/replication/inference.sh via:
    eval "$(python scripts/print_winner_exports.py <winners.json> <arch>)"
"""

import json
import shlex
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: print_winner_exports.py <winners.json> <arch>")
    winners_path, arch = sys.argv[1], sys.argv[2]
    with open(winners_path) as f:
        winners = json.load(f)
    if arch not in winners:
        sys.exit(f"ERROR: {arch} not in {winners_path}")
    w = winners[arch]
    print(f"WINNER_TYPE={shlex.quote(w['type'])}")
    print(f"WINNER_PATH={shlex.quote(w['path'])}")
    print(f"WINNER_SCALER={shlex.quote(w.get('scaler_path', ''))}")
    print(f"BASE_MODEL={shlex.quote(w['base_model'])}")


if __name__ == "__main__":
    main()
