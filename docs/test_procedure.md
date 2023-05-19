# Testing #

- Run `python -m tests.generate_tests` to add all test simulation yaml files to `test_data/` directory.
- Run `python -m tests.run_tests` to run all test configurations.
- Those that run or crash will be in the logs.
- Record simulation crashes as C.
- Record tests that ran as P (note these still could have garbage output).
- Collect this info in the `docs/test_results.md` file using the following template for each simulation.

---

Name: example

Output looked at: None

Result (P/C): P

---
