# attention-memory
Copied essential files from [CleanRL(94a44b5)](https://github.com/vwxyzjn/cleanrl/commit/94a44b5a252c432e3c47577fa46ed49c230fcce3).
# Get Started
Prerequisites:
- Python 3.9
- [Poetry 1.2.1+](https://python-poetry.org/)

We recommand to use Anaconda(or Miniconda) to run in virtual environment.

create the virtual environment.
```bash
conda create -n attention-memory python=3.9 -y
conda activate attention-memory
```

Clone the repo:
```bash
git clone https://github.com/jinPrelude/attention-memory.git
cd attention-memory
```

Install poetry and run `poetry install`. This will install all dependencies we need.
```bash

pip install poetry && poetry install
```


If the installation completed, try ballet_lstm_lang_only.py
```bash
# --track for logging wandb.ai
python ballet_lstm_lang_only --track
```