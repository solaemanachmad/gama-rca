.PHONY: test lint
test:        ## prompt-equivalence check for every retained arm (no data or model needed)
	@for f in tests/snapshots/*.json; do python tests/prompt_snapshot.py compare $$f | tail -1; done
	@python tests/test_fingerprint.py | tail -1
	@python tests/test_calibrated.py | tail -1
	@python tests/test_retype.py | tail -1
	@python tests/test_hypothesis_rca.py | tail -1
	@python tests/test_causal.py | tail -1
	@python tests/test_belief.py | tail -1
	@python tests/test_logpre.py | tail -1
lint:        ## unused imports / variables
	python -m pyflakes $$(git ls-files '*.py')
