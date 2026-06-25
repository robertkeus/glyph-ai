#!/usr/bin/env sh
# Full local gate — zero dependencies, any python3. Must be green before any model run.
set -e
python3 -m unittest discover -s tests -t .   # correctness (25 tests)
python3 smoke.py                             # verifier soundness diagnostic
python3 -m glyph.curriculum                  # curriculum shape + split novelty
