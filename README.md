# attention-memory
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
Implementation of the paper [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039) using [CleanRL](https://github.com/vwxyzjn/cleanrl/commit/94a44b5a252c432e3c47577fa46ed49c230fcce3).

### Progress
- [x] [gym-style BalletEnv reimplementation](https://github.com/jinPrelude/gym-balletenv)
- [x] [LSTM Agent](https://github.com/jinPrelude/attention-memory/blob/gymnasium/lstm_ballet.py)
- [ ] TrXL Agent
- [ ] HCAM Agent

# Get Started
We recommand to use Anaconda(or Miniconda) to run in virtual environment.
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

# Benchmark

### LSTM Agent
Tested on i9-11900k + RTX 3090 :
| Playing BalletEnv (2_delay16_easy) |
|:---:|
|<img src="https://user-images.githubusercontent.com/16518993/216736601-3099e3c1-f734-4078-a87c-30eeba5e0310.gif" width="300" height="150"/>|

| Trained in ≈15M total frames | Trained in ≈3H |
|:---:|:---:|
|![](https://user-images.githubusercontent.com/16518993/216736884-3a897014-447c-4780-8ef1-ca164f6e0179.png)|![](https://user-images.githubusercontent.com/16518993/216736739-394388c8-792a-4c2e-aff6-6c3f6099e374.png)|

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://jinprelude.github.io"><img src="https://avatars.githubusercontent.com/u/16518993?v=4?s=100" width="100px;" alt="Euijin Jeong"/><br /><sub><b>Euijin Jeong</b></sub></a><br /><a href="https://github.com/jinPrelude/attention-memory/commits?author=jinPrelude" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!