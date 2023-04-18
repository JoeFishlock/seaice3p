# Testing #

## Procedure ##

- Run generate_test_configurations.py to add all test simulation yaml files to `test/data` directory.
- Run run_tests.py to run all test configurations.
- Those that run or crash will be in the logs.
- Record simulation crashes as C.
- Record tests that ran as P (note these still could have garbage output).
- Collect this info in the results section using the following template for each simulation.

---

Name: example

Output looked at: None

Result (P/C): P

---

## Results ##

Version: 0.3.0

2023-04-17 23:03

---

Name: ConstantMicroLXF

Output looked at: None

Result (P/C): P

---

Name: YearlyMacroLU

Output looked at: None

Result (P/C): P

---

Name: ConstantMicroLU

Output looked at: None

Result (P/C): P

---

Name: YearlyMicroLXF

Output looked at: None

Result (P/C): P

---

Name: YearlyMedLU

Output looked at: None

Result (P/C): P

---

Name: ConstantMedLU

Output looked at: None

Result (P/C): P

---

Name: ConstantMacroLU

Output looked at: None

Result (P/C): P

---

Name: YearlyMicroLU

Output looked at: None

Result (P/C): P

---

Name: ConstantMedLXF

Output looked at: None

Result (P/C): P

---

Name: YearlyMacroLXF

Output looked at: None

Result (P/C): C

---

Name: YearlyMedLXF

Output looked at: None

Result (P/C): C

---

Name: ConstantMacroLXF

Output looked at: None

Result (P/C): C

---
