#!/usr/bin/env bash

# Create a ZIP archive of the project, excluding Git metadata,
# virtual environments, build artifacts, Python cache files,
# and the StableHLO submodule.
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
     "external/stablehlo/*" \
     "*/__pycache__/*" \
     "*.pyc"

echo "Projeto compactado em: $OUTPUT"