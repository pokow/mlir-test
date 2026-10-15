#!/usr/bin/env bash

# Create a ZIP archive of the project, excluding unnecessary stuff
#
# Usage:
#   bash scripts/zip.sh
#
# Output:
#   ../mlir-test.zip

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
OUTPUT="$(dirname "$PROJECT_DIR")/${PROJECT_NAME}.zip"

cd "$PROJECT_DIR"

# Remove the previous archive, if it exists.
rm -f "$OUTPUT"

zip -r "$OUTPUT" . \
  -x "*/.git/*" ".git/*" \
     "*/.venv/*" ".venv/*" \
     "*/bin/*" "bin/*" \
     "*/build/*" "build/*" \
     "*/.vscode/*" ".vscode/*" \
     "*/.cache/*" ".cache/*" \
     ".clangd" \
     "external/stablehlo/*" \
     "*/__pycache__/*" \
     "*.pyc"

echo "Project zipped to: $OUTPUT"