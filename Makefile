
init:
	mkdir models
	mkdir models/agent-0
	mkdir models/agent-1
	pip install -r requirements.txt

test:
	pytest --cov=maddpg/ tests/

mypy:
	mypy maddpg

lint:
	black maddpg/
	pylint maddpg