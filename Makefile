MSG = "Default commit message"


setup: # install pre-commit hooks
	pip install pre-commit black isort pylint
	pre-commit install
	pre-commit run --all-files

build: # build docker
	...

run: # run docker
	...

build-run: # build and run docker
	...

commit: # commit changes
	pre-commit run --all-files
	git add .
	git commit -m "$(MSG)"

push: # push changes
	git push

commit-push: # commit and push changes
	make commit
	make push