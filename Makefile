.PHONY: test coverage coverage-html clean

test:
	pytest -v

coverage:
	coverage run -m pytest
	coverage report

coverage-html:
	coverage run -m pytest
	coverage html
	open htmlcov/index.html

clean:
	rm -rf __pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/