# Contributing to EPM

Thanks for taking an interest in EPM. This is a research project about
**efficient predictive multimodal generative modelling**, and contributions are
welcome — from a typo fix in the docs to a new benchmark dataset or a
replication of the Stage 13 H1 experiment on different hardware.

## Ground rules

EPM has one rule that outranks the others: **measured results only.** No number
enters the README, the docs or a report unless it came out of a real run that
someone else can reproduce. If you cannot reproduce a claim, that is a bug
worth filing.

## Ways to contribute

- **Replicate an experiment.** Run `python demo_stage13.py` on your hardware and
  open an issue with your numbers, seeds and machine spec. Divergence from the
  recorded findings is interesting, not unwelcome.
- **Add a dataset** to the registry (`src/datasets/`). Every entry must document
  its source and licence, and pass `python run.py --verify-datasets`.
- **Add or improve an algorithm** so it is benchmarkable through the fair
  benchmark runner, sharing the same split and seed as everything else.
- **Improve the docs** in `docs/`. The chapters are written to be read in order;
  corrections to the mathematics are especially valuable.
- **Report a bug** — leakage, a broken split, a metric computed the wrong way,
  a result that will not reproduce.

## Development setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
pytest tests/ -q               # the full suite must pass
black .                        # formatting (line length 88)
isort .                        # import order
```

If your change touches a stage, run that stage's demo (`python demo_stage<N>.py`)
and say in the PR what you observed.

## Pull request expectations

- One logical change per PR. A refactor and a new feature should be two PRs.
- Explain **what you measured**, not only what you wrote. A PR that claims a
  speed-up should include the before and after numbers and the command used.
- New behaviour needs a test. The suite is the project's main defence against
  silent breakage in the preprocessing and metric code.
- Update the relevant `docs/` chapter if you changed how something works.

## Reporting a security issue

Please do not open a public issue for a security problem. See
[SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
