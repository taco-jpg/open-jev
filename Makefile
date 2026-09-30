.PHONY: test lint smoke

test:
	pytest -q

lint:
	ruff check .

smoke:
	open-jev prepare examples/tiny.jsonl /tmp/open-jev-smoke --validation-fraction 0

