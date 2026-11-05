.PHONY: test lint
test:        ## unit + golden-snapshot tests (no data or model needed)
	@python tests/test_fingerprint.py | tail -1
	@python tests/test_calibrated.py | tail -1
	@python tests/test_logpre.py | tail -1
	@python tests/test_gama.py | tail -1
	@python tests/test_aiops2025.py | tail -1
	@python tests/test_api_client.py | tail -1
lint:        ## unused imports / variables (active code only)
	python -m pyflakes $$(git ls-files '*.py' | grep -v '^archive/')
