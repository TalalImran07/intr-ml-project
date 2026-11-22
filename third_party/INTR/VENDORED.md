# Vendored copy of the official INTR code

Source: https://github.com/Imageomics/INTR
Commit: 69b8a693d774b4805efe1adc1d0c4c4a28200568 (Mon Apr 15 2024)
License: Apache 2.0 (see LICENSE in this folder)

Only the Python packages needed to build and train the model were copied
(`datasets/`, `models/`, `util/`, `engine.py`, `main.py`). These files are
**unmodified**. Everything our project changes is done from outside, in
`intr_pet/` and `train.py`, so this folder stays identical to the authors' code.
