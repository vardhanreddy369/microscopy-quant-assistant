#!/usr/bin/env bash
# Start the demo. Handles the virtualenv so there is nothing to remember.
#
#   ./run_demo.sh
#
# Creates the environment on first run, then launches the app in your browser.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/streamlit ]; then
  echo "Setting up the environment (first run only, about a minute)..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
  echo "Done."
fi

# Fail loudly here rather than showing a broken page to the professor.
if [ ! -f sample_data/public_human_mitosis.png ]; then
  echo "ERROR: sample images are missing."
  echo "Run: .venv/bin/python scripts/make_sample_data.py"
  exit 1
fi

echo
echo "  Starting the Microscopy Quantification Assistant"
echo "  A browser tab will open at http://localhost:8501"
echo "  Press Ctrl+C in this window when you are finished."
echo

exec .venv/bin/streamlit run app.py
