DOCKER_IMG = slava-base-image
DOCKER_CONTAINER = slava-base-container
DATA_DIR = /SSD/slava
ENV_FILE = env.env
MSG = "Default commit message"


setup: # install pre-commit hooks
	pip install pre-commit black isort pylint
	pre-commit install
	pre-commit run --all-files

build: # build docker
	docker build -t "$(DOCKER_IMG)" .

run: # run docker
	docker run -d --gpus all -v ".:/code" -v "$(DATA_DIR):/mnt" --env-file "$(ENV_FILE)" \
	--shm-size=8gb -p 8080:8080 --name "$(DOCKER_CONTAINER)" "$(DOCKER_IMG)" tail -f /dev/null

exec:
	docker exec -it "$(DOCKER_CONTAINER)" /bin/bash

stop: # stop docker
	docker stop "$(DOCKER_CONTAINER)"

remove: # remove docker
	docker rm "$(DOCKER_CONTAINER)"

build-run: build run

commit: # commit changes
	pre-commit run --all-files
	git add .
	git commit -m "$(MSG)"

push: # push changes
	git push

commit-push: commit push