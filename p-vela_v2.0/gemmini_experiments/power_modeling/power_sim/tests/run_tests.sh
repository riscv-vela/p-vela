#!/usr/bin/env bash
# Run every unit test and report a single pass/fail.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/unit"
fail=0
for t in test_*.py; do
    printf '%-40s ' "${t}"
    if out=$(python3 "${t}" 2>&1); then
        printf 'PASS  (%s)\n' "$(grep -o 'Ran [0-9]* test[s]*' <<<"${out}")"
    else
        printf 'FAIL\n'; echo "${out}" | sed 's/^/    /'; fail=1
    fi
done
exit "${fail}"
