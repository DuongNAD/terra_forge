#!/usr/bin/env bash
# ==============================================================================
# TerraForge: One-Click Automated Setup & Installation Script
# Compatible with macOS and Linux
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "        🏔️  TerraForge World Generator - Automated Setup"
echo "================================================================="

# 1. Check Python version
if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
else
    echo "❌ Error: Python 3 not found. Please install Python 3.10+."
    exit 1
fi

PY_VER=$($PY_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  [1/4] Found Python: $PY_BIN (version $PY_VER)"

# 2. Install dependencies & TerraForge CLI in editable mode
echo "  [2/4] Installing dependencies & 'terraforge' CLI..."
$PY_BIN -m pip install -q -e .

# 3. Locate & Install Blender Add-on
echo "  [3/4] Installing TerraForge GUI add-on into Blender..."
if command -v terraforge >/dev/null 2>&1; then
    terraforge install-addon
else
    $PY_BIN -m terra_forge.cli install-addon
fi

# 4. Run Doctor diagnostic
echo "  [4/4] Running system healthcheck..."
if command -v terraforge >/dev/null 2>&1; then
    terraforge doctor
else
    $PY_BIN -m terra_forge.cli doctor
fi

echo "================================================================="
echo "🎉 Setup complete! You can now run:"
echo ""
echo "   terraforge presets              # View ready-to-use map presets"
echo "   terraforge generate -p alpine_fjord -r  # Generate 3D diorama + preview"
echo "   terraforge doctor               # Check environment status"
echo "================================================================="
