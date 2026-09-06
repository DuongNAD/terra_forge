# ==============================================================================
# TerraForge Makefile
# ==============================================================================

.PHONY: help install doctor presets generate test clean

help:
	@echo "TerraForge Management Commands:"
	@echo "  make install     - One-click install dependencies, CLI, and Blender add-on"
	@echo "  make doctor      - Run full environment & system health check"
	@echo "  make presets     - List available 3D landscape presets"
	@echo "  make generate    - Generate default alpine fjord diorama with 2x2 preview"
	@echo "  make test        - Test generation on all presets (alpine, volcanic, karst)"
	@echo "  make clean       - Remove cache files and build artifacts"

install:
	./install.sh

doctor:
	terraforge doctor

presets:
	terraforge presets

generate:
	terraforge generate -p alpine_fjord -r -o models

test:
	terraforge generate -p alpine_fjord -r -o models
	terraforge generate -p volcanic_marsh -r -o models
	terraforge generate -p karst_cavern -r -o models
	@echo "✅ All presets generated and validated successfully!"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleaned cache and build directories."
