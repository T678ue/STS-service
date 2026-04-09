.PHONY: proto install dev test lint run run-dev clean

# Proto compilation — generates Python stubs from .proto files
PROTO_SRC = proto/sts/v1/service.proto
PROTO_OUT = sts/transport/_generated

proto:
	mkdir -p $(PROTO_OUT)
	python -m grpc_tools.protoc \
		--proto_path=proto \
		--python_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		--pyi_out=$(PROTO_OUT) \
		$(PROTO_SRC)
	touch $(PROTO_OUT)/__init__.py
	@echo "Proto stubs generated in $(PROTO_OUT)/"

# Install dependencies
install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

# Run
run:
	uv run python -m sts

run-dev:
	uv run python -m sts --dev

# Test
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=sts --cov-report=term-missing

# Lint
lint:
	uv run ruff check sts/ tests/
	uv run mypy sts/

format:
	uv run ruff check --fix sts/ tests/
	uv run ruff format sts/ tests/

# Container
build:
	podman build -t sts-service -f Containerfile .

run-container:
	podman run --rm -p 8765:8765 -p 50051:50051 -p 9090:9090 sts-service

run-container-gpu:
	podman run --rm --device nvidia.com/gpu=all \
		-p 8765:8765 -p 50051:50051 -p 9090:9090 sts-service

# Clean
clean:
	rm -rf $(PROTO_OUT) dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
