# Kaggle submission

## Quick path (prove the pipeline)

1. Open the competition **TransferPolicy Notebook** on the Kaggle Code tab and copy it.
2. Paste the contents of [`kaggle_submission.py`](kaggle_submission.py) into the notebook (or attach this file as a dataset and `%run` it).
3. Ensure the competition dataset is attached (provides `kaggle_evaluation`).
4. **Save Version → Submit** to the competition.

First pipeline prove-out: change the server line to use `PIDWaterResetPolicy` instead of `KaggleTransferPolicy` if you want a known baseline on the LB.

## Local rebuild

```bash
python scripts/build_kaggle_notebook.py
```

## Constraints

- 9 hour notebook, no internet
- 15 min first step / scenario (model load), 60 s / step
- Fresh policy per scenario — no cross-scenario state
- Actions must be finite and in bounds
