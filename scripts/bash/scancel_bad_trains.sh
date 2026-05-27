#!/usr/bin/env bash

# Usage:
#   ./scancel_bad_trains.sh --dry-run Lorenz
#   ./scancel_bad_trains.sh Lorenz
#
# Options:
#   -n, --dry-run   Show matching jobs but do not cancel them.
#   -h, --help      Show usage.
#
# The pattern is matched as a substring against the Slurm job NAME column.

set -euo pipefail

dry_run=false
pattern="Lorenz"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-n|--dry-run)
			dry_run=true
			shift
			;;
		-h|--help)
			echo "Usage: $0 [--dry-run|-n] [job-name-pattern]"
			exit 0
			;;
		*)
			pattern="$1"
			shift
			;;
	esac
done

if [[ -z "${pattern}" ]]; then
	echo "Usage: $0 [--dry-run|-n] [job-name-pattern]" >&2
	exit 1
fi

mapfile -t job_ids < <(
	squeue -h -u "${USER}" -o "%A|%j" | awk -F '|' -v pattern="${pattern}" 'index($2, pattern) {print $1}'
)

if [[ ${#job_ids[@]} -eq 0 ]]; then
	echo "No running jobs matched pattern: ${pattern}"
	exit 0
fi

echo "Matching jobs:"
squeue -h -u "${USER}" -o "%A|%j|%T|%M|%R" | awk -F '|' -v pattern="${pattern}" 'index($2, pattern) {print}' | tr '|' '\t'

if [[ "${dry_run}" == true ]]; then
	echo
	echo "Dry run enabled; no jobs were cancelled."
	exit 0
fi

echo
echo "Cancelling job IDs: ${job_ids[*]}"
scancel "${job_ids[@]}"
