.DEFAULT_GOAL := help

DOCKER  ?= sudo docker
# The compose file lives in docker/, so point it at the root .env explicitly
COMPOSE  = $(DOCKER) compose --env-file .env -f docker/docker-compose.yml
SERVICE  = asr_serve

# Load variables from .env so targets can use them (e.g. RAY_FASTAPI_PORT in `health`).
-include .env
export

RAY_FASTAPI_PORT   ?= 8000
RAY_DASHBOARD_PORT ?= 8265

.PHONY: help env build prefetch up start down restart logs ps health clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.sample if it does not exist
	@if [ ! -f .env ]; then \
		cp .env.sample .env; \
		echo "Created .env from .env.sample - edit it to customize settings"; \
	else \
		echo ".env already exists, skipping"; \
	fi

build: env ## Build the service image
	$(COMPOSE) build

prefetch: build ## Download the configured model into the HuggingFace cache
	$(COMPOSE) run --rm --no-deps -e HF_HUB_OFFLINE=0 $(SERVICE) python -m app.prefetch

up: env ## Build if needed and start the service in the background
	$(COMPOSE) up -d --build --remove-orphans
	@echo "NeMo ASR service is starting at http://localhost:$(RAY_FASTAPI_PORT)"
	@echo "Ray dashboard: http://localhost:$(RAY_DASHBOARD_PORT)"
	@echo "Follow logs with: make logs"

start: env ## Start the service in the foreground
	$(COMPOSE) up --build --remove-orphans

down: ## Stop and remove the service container
	$(COMPOSE) down

restart: down up ## Restart the service

logs: ## Follow service logs
	$(COMPOSE) logs -f $(SERVICE)

ps: ## Show service status
	$(COMPOSE) ps

health: ## Check that the Serve proxy is healthy (/-/healthz)
	@curl -sf -o /dev/null http://localhost:$(RAY_FASTAPI_PORT)/-/healthz && echo "OK" || (echo "Service is not healthy" && exit 1)

clean: ## Stop the service and remove its containers, networks and images
	$(COMPOSE) down --rmi local