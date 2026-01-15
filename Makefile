.PHONY: setup frontend backend dev clean docker modal-deploy

# Install all dependencies
setup:
	cd frontend && npm install
	cd backend && pip install -r requirements.txt
	cd backend && python generate_assets.py

# Run frontend dev server
frontend:
	cd frontend && npm run dev

# Run backend dev server
backend:
	cd backend && uvicorn app.main:app --reload --port 8000

# Run both in parallel (requires terminal multiplexer or two terminals)
dev:
	@echo "Run 'make frontend' and 'make backend' in separate terminals"
	@echo "Or use 'docker-compose up' for containerized development"

# Clean build artifacts
clean:
	rm -rf frontend/node_modules frontend/dist
	rm -rf backend/__pycache__ backend/app/__pycache__
	rm -rf backend/*.egg-info

# Docker development
docker:
	docker-compose up --build

# Deploy Modal inference service
modal-deploy:
	cd backend && modal deploy modal_inference.py

# Run Modal inference locally (for testing)
modal-test:
	cd backend && modal run modal_inference.py
