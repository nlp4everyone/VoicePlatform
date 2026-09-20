.DEFAULT_GOAL := help

DOCKER  ?= sudo docker
COMPOSE  = $(DOCKER) compose
SERVICE  = omnivoice

# Load variables from .env so targets can use them (e.g. VLLM_PORT in `health`).
-include .env
export

VLLM_PORT ?= 8002

.PHONY: help env build up start down restart logs ps health clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.sample if it does not exist
	@if [ ! -f .env ]; then \
		cp .env.sample .env; \
		echo "Created .env from .env.sample - edit it to customize settings"; \
	else \
		echo ".env already exists, skipping"; \
	fi

build: env ## Build the service image (audio deps are baked in once)
	$(COMPOSE) build

up: env ## Build if needed and start the service in the background
	$(COMPOSE) up -d --build
	@echo "KhanhTTS-OmniVoice service is starting at http://localhost:$(VLLM_PORT)"
	@echo "Follow logs with: make logs"

start: env ## Start the service in the foreground
	$(COMPOSE) up --build

down: ## Stop and remove the service container
	$(COMPOSE) down

restart: down up ## Restart the service

logs: ## Follow service logs
	$(COMPOSE) logs -f $(SERVICE)

ps: ## Show service status
	$(COMPOSE) ps

health: ## Check the service health endpoint
	@curl -sf http://localhost:$(VLLM_PORT)/health && echo "OK" || (echo "Service is not healthy" && exit 1)

clean: ## Stop the service and remove its containers, networks and images
	$(COMPOSE) down --rmi local