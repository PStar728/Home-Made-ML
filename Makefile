.PHONY: all train clean test run help init-log

# Default target
all: train

# Initialize logging (create ML_log.xlsx with Temp Dump sheet)
init-log:
	python3 init_log.py

# Train the model
train: init-log
	python3 main.py

# Clean generated files
clean:
	rm -f weights.json Temp_Holder.txt ML_log.xlsx

# Test the model (placeholder - no tests implemented yet)
test:
	@echo "No tests implemented yet"

# Run the model (same as train for now)
run: train

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Train the model (default)"
	@echo "  train      - Train the model (initializes logging first)"
	@echo "  init-log   - Create ML_log.xlsx with Temp Dump sheet"
	@echo "  clean      - Remove generated files (weights.json, Temp_Holder.txt, ML_log.xlsx)"
	@echo "  test       - Run tests (not implemented)"
	@echo "  run        - Run the model (same as train)"
	@echo "  help       - Show this help message"