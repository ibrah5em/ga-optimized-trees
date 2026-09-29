# GECCO paper

`main.tex` is an ACM `acmart` (sigconf) paper in double-blind review mode.

## Build

No number in `main.tex` is typed by hand. Regenerate the numbers, tables and figures from
the committed evidence first:

```bash
python scripts/paper_assets.py      # writes paper/gecco/generated/
```

Then compile with any TeX Live that includes `acmart` (Overleaf works: upload this
directory, including `generated/`):

```bash
cd paper/gecco && latexmk -pdf main.tex
```

A missing evidence set leaves its macros undefined, so LaTeX fails instead of printing a
stale number.

## Before submission

- Check the current GECCO call for the page limit, template version and anonymity rules.
- The repository link must be anonymised for review (e.g. anonymous.4open.science).
- `refs.bib`: every entry with a DOI was checked against Crossref on 2026-09-29 (title,
  volume, issue, pages, year). Entries without a DOI (JMLR, PMLR, NeurIPS, MLSys, the
  Holm and CART classics) were not machine-checked; confirm them before submission.
- Authors: see the authorship note in `paper/PLAN.md` (Phase 5).
