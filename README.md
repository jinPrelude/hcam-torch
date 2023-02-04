# attention-memory
Reproduction of the paper [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039) using [CleanRL](https://github.com/vwxyzjn/cleanrl/commit/94a44b5a252c432e3c47577fa46ed49c230fcce3).

# Progress
- [x] [BalletEnv gym-style reimplementation](https://github.com/jinPrelude/gym-balletenv)
- [x] [LSTM Agent](https://github.com/jinPrelude/attention-memory/blob/gymnasium/lstm_ballet.py)
- [ ] TrXL Agent
- [ ] HCAM Agent

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