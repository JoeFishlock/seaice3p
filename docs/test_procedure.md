# Testing #

- Run generate_test_configurations.py to add all test simulation yaml files to `test/data` directory.
- Run run_tests.py to run all test configurations.
- Those that run or crash will be in the logs.
- Record simulation crashes as C.
- Record tests that ran as P (note these still could have garbage output).
- Collect this info in the `docs/test_results.md` file using the following template for each simulation.

---

Name: example

Output looked at: None

Result (P/C): P

---
