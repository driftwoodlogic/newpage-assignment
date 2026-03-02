.PHONY: up down logs promptfoo-eval promptfoo-view

up:
	docker compose up -d db phoenix

down:
	docker compose down

logs:
	docker compose logs -f db phoenix

promptfoo-eval:
	PROMPTFOO_CONFIG_DIR=.promptfoo promptfoo eval -c eval/promptfoo.yaml

promptfoo-view:
	PROMPTFOO_CONFIG_DIR=.promptfoo promptfoo view
