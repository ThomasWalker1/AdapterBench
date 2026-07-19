# AdapterBench leaderboards

One leaderboard per setting. Each **entry is a shape (codec) together with the hyperparameters
that produced its number**, and every entry is reproducible: run the recorded command against the
codec in this repository (`src/adapterbench/t2p/codecs.py` + its manifest in `configs/adapters/`)
and you regenerate the result up to seed variance.

There is no automated search — hyperparameters are chosen per entry (by hand or by a sweep the
author runs) and recorded here alongside the number. What keeps the comparison a *shape* comparison
rather than a hyperparameter contest is a small set of fixed rules, not machinery:

- **Report `matched − control`, never a raw loss.** Every setting has a control a non-conditioning
  adapter cannot pass; the headline number is matched minus control. Raw loss is not capability (a
  response cross-entropy can hit zero with task success at chance) — but a *controlled* loss
  difference is fair: the task setting's `matched − static` subtracts a same-shape reference that
  captures any generic loss reduction, and reports generation accuracy alongside it.
- **Only the free optimization HPs vary per entry.** The *free* HPs are scale, learning rate,
  warmup, step budget (and, for the image setting, the noise regularization weight `λ`). The
  *shared substrate* — task data, conditioner/trunk, evaluator, and the control — is identical
  across entries and lives in the setting's code; it is never tuned per entry. The *shape-identity*
  HP (LoRA's rank) is fixed, not maximized. An entry that changes the substrate is not comparable.
- **Multi-seed.** The measured quantity is stochastic; report ≥3 seeds (mean ± spread).
- **Scale is swept.** Report the entry at its best scale (the sweep is the author's; the winning
  scale is what appears here), because the same shape can fail at one scale and succeed at another.

Columns: **shape**, the free HPs, **matched − control** (mean ± std over seeds), **# seeds**, and
the exact **reproduce** command. The current baseline commit is noted per file; re-running an entry
under `main` reproduces it. Losers are kept, not curated away — the "does shape matter?" question
needs the negative results.

To add an entry: land the codec (subclass + `make_codec` entry + `configs/adapters/<name>.yaml`),
run the setting's CLI with your chosen free HPs over ≥3 seeds, and append a row with the number and
the command that produced it.
