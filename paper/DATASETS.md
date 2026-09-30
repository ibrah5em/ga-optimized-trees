# Benchmark Datasets

**Fixed 2026-08-07, before any Phase 3 run.** `paper/PREREGISTRATION.md` requires the
dataset list be chosen before results are seen. This is that list.

Do not add, drop or substitute a dataset after the first reported run. If one becomes
unusable, record it in the Deviations table below with the reason, and report results both
with and without it.

______________________________________________________________________

## Selection rule, stated before selection

1. Member of **OpenML-CC18** (study 99), verified against the live OpenML API — not
   assumed from a name or a remembered ID.
1. Classification task.
1. Between 500 and 1500 rows. The lower bound keeps 10-fold × 3-repeat CV meaningful; the
   upper bound keeps a nested run with a GA inside it computationally feasible.
1. At most ~50 features. A tree over 500 features is not an interpretable model, so
   including such datasets would not test the claim being made.
1. Smallest class of at least 40 members, so stratified 10-fold CV is valid without
   silently reducing the fold count.

Applying the rule to CC-18 yields exactly the 20 datasets below. `spambase` (4601 rows),
`phoneme` (5404), `car` (1728) and `cmc` (1473) met criteria 1, 2, 4 and 5 but fell outside
the size window; they were excluded by the rule, not by inspection of any result.

**iris, wine and breast_cancer are not on this list.** They are saturated, and they carried
every withdrawn claim in `paper/CLAIMS.md`. `wdbc` (ID 1510) is the same Wisconsin
diagnostic data as sklearn's `breast_cancer`, so that problem is still represented, but it
now arrives through the pre-registered CC-18 path.

______________________________________________________________________

## The 20 datasets

Every row was fetched from OpenML on 2026-08-07; `n`, `p`, classes and smallest-class size
are measured, not quoted.

| Loader name         | OpenML ID | OpenML name                      |    n |   p | Classes | Smallest class |
| ------------------- | --------- | -------------------------------- | ---: | --: | ------: | -------------: |
| `dresses_sales`     | 23381     | dresses-sales                    |  500 |  12 |       2 |            210 |
| `kc2`               | 1063      | kc2                              |  522 |  21 |       2 |            107 |
| `climate_crashes`   | 40994     | climate-model-simulation-crashes |  540 |  18 |       2 |             46 |
| `wdbc`              | 1510      | wdbc                             |  569 |  30 |       2 |            212 |
| `ilpd`              | 1480      | ilpd                             |  583 |  10 |       2 |            167 |
| `balance_scale`     | 11        | balance-scale                    |  625 |   4 |       3 |             49 |
| `credit_approval`   | 29        | credit-approval                  |  690 |  15 |       2 |            307 |
| `breast_w`          | 15        | breast-w                         |  699 |   9 |       2 |            241 |
| `eucalyptus`        | 188       | eucalyptus                       |  736 |  19 |       5 |            105 |
| `blood_transfusion` | 1464      | blood-transfusion-service-center |  748 |   4 |       2 |            178 |
| `diabetes_pima`     | 37        | diabetes                         |  768 |   8 |       2 |            268 |
| `analcatdata_dmft`  | 469       | analcatdata_dmft                 |  797 |   4 |       6 |            123 |
| `vehicle`           | 54        | vehicle                          |  846 |  18 |       4 |            199 |
| `tic_tac_toe`       | 50        | tic-tac-toe                      |  958 |   9 |       2 |            332 |
| `vowel`             | 307       | vowel                            |  990 |  12 |      11 |             90 |
| `credit_g`          | 31        | credit-g                         | 1000 |  20 |       2 |            300 |
| `qsar_biodeg`       | 1494      | qsar-biodeg                      | 1055 |  41 |       2 |            356 |
| `pc1`               | 1068      | pc1                              | 1109 |  21 |       2 |             77 |
| `banknote`          | 1462      | banknote-authentication          | 1372 |   4 |       2 |            610 |
| `pc4`               | 1049      | pc4                              | 1458 |  37 |       2 |            178 |

Span: 500–1458 rows, 4–41 features, 2–11 classes. Available in code as
`DatasetLoader.CC18_BENCHMARK`.

______________________________________________________________________

## Two loader entries were serving the wrong data

Found while verifying the IDs above, and worth recording because both had been usable from
the CLI for the whole life of the project:

| Name           | Was    | Actually is                    | Now                    |
| -------------- | ------ | ------------------------------ | ---------------------- |
| `heart`        | ID 4   | `labor` — 57 rows, 16 features | ID 53, `heart-statlog` |
| `mammographic` | ID 310 | `mammography` — 11183 rows     | renamed `mammography`  |

`python scripts/train.py --dataset heart` trained on labor-relations data and reported it
as heart disease. No published claim depended on it — `paper/CLAIMS.md` traces every
withdrawn number to iris, wine or breast_cancer — but the same class of error is exactly
what this audit exists to catch, and it would have reached the paper had `heart` been
included in a run.

Neither dataset is in CC-18, so neither is part of the benchmark above.

______________________________________________________________________

## Deviations

Record any departure from this list here, with a date and reason, **before** the affected
result is used.

| Date | Deviation | Reason |
| ---- | --------- | ------ |
| —    | —         | —      |
