.DEFAULT_GOAL := help

DOCKER  ?= sudo docker
COMPOSE  = $(DOCKER) compose
SERVICE  = fun-asr-mlt
CONVERT  = fun-asr-convert

# Converted vLLM model; its weights file is written last (via a temp name), so it only
# exists once a conversion has fully finished and doubles as the "model ready" marker.
MODEL_DIR     = models/Fun-ASR-MLT-Nano-2512-vllm
MODEL_WEIGHTS = $(MODEL_DIR)/model.safetensors

# Load variables from .env so targets can use them (e.g. VLLM_PORT in `health`).
-include .env
export

VLLM_PORT ?= 8002

.PHONY: help env build model model-clean up start down restart logs ps health clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if it does not exist
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example - edit it to customize settings"; \
	else \
		echo ".env already exists, skipping"; \
	fi

build: env ## Build the service image (audio deps are baked in once)
	$(COMPOSE) build

# Only runs when the converted weights are missing: the first run downloads the checkpoint
# into HF_CACHE_DIR and converts it, later runs skip this recipe entirely.
$(MODEL_WEIGHTS): | build
	$(COMPOSE) run --rm $(CONVERT)

model: $(MODEL_WEIGHTS) ## Download and convert the model (no-op once it is converted)

model-clean: ## Delete the converted model (re-converting reuses HF_CACHE_DIR, no re-download)
	$(COMPOSE) run --rm --entrypoint rm $(CONVERT) -rf /$(MODEL_DIR)

up: model ## Prepare the model if needed and start the service in the background
	$(COMPOSE) up -d --no-deps $(SERVICE)
	@echo "Fun-ASR service is starting at http://localhost:$(VLLM_PORT)"
	@echo "Follow logs with: make logs"

start: model ## Prepare the model if needed and start the service in the foreground
	$(COMPOSE) up --no-deps $(SERVICE)

down: ## Stop and remove the service containers
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
