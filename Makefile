.DEFAULT_GOAL := help

.PHONY: build test bench race cover lint fmt check help

build: ## Build all packages
	go build ./...

test: ## Run all tests
	go test ./...

bench: ## Run benchmarks
	go test -bench=. -benchmem ./...

race: ## Run tests with race detector
	go test -race ./...

cover: ## Generate HTML coverage report
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

lint: ## Run linter
	golangci-lint run

fmt: ## Format all Go files
	gofmt -w .

check: lint test race ## Run lint + test + race (full CI check)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
