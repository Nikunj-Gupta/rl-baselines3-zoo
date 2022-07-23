LINT_PATHS = *.py tests/ scripts/ utils/

# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent --color=yes

# Type check
type:
	pytype -j auto ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics


format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format type lint

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

.PHONY: docker lint type pytest


#test_custom 
run: 
	python3 train.py \
	--algo ppo \
	--env CartPole-v1 \
	-n 50000 \
	-optimize \
	--sampler skopt --pruner halving \
	--optimization-log-path summaries/CartPole-v1/ \
	--tensorboard-log summaries \
	--n-trials 1000 \
	--n-jobs 2 \
	--eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 \
	--save-freq 10000 


req: 
	pip3 install -r requirements.txt 

dist_run: 
	python3 train.py --algo ppo --env MountainCar-v0 -optimize --study-name test --storage sqlite:///example.db

save_hyperparameters: 
	python scripts/parse_study.py -i logs/ppo/report_MountainCar-v0_500-trials-1000000-tpe-median_1658558124.pkl --print-n-best-trials 10 --save-n-best-hyperparameters 10 