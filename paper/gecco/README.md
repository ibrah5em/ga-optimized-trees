# GECCO paper

`main.tex` is an ACM `acmart` (sigconf) paper in double-blind review mode.

## Build

No number in `main.tex` is typed by hand. Regenerate the numbers, tables and figures from
the committed evidence first:

```bash
python scripts/paper_assets.py      # writes paper/gecco/generated/
```

Then compile with any TeX Live that includes `acmart` (Overleaf works: upload this
directory, including `generated/`), or with [tectonic](https://tectonic-typesetting.github.io),
which fetches what it needs on first run:

```bash
cd paper/gecco && latexmk -pdf main.tex      # or: tectonic -X compile main.tex
```

`main.pdf` is the committed build (tectonic 0.15, 2026-09-29): 6 pages in review mode, no
overfull lines. Rebuild it after any change to `paper/evidence/` or `main.tex`.

A missing evidence set leaves its macros undefined, so LaTeX fails instead of printing a
stale number.

## Before submission

- Check the current GECCO call for the page limit (the draft is 6 pages including
  references), template version, track and anonymity rules.
- The repository link must be anonymised for review (e.g. anonymous.4open.science).
- `refs.bib`: every entry with a DOI was checked against Crossref on 2026-09-29 (title,
  volume, issue, pages, year). Entries without a DOI (JMLR, PMLR, NeurIPS, MLSys, the
  Holm and CART classics) were not machine-checked; confirm them before submission.
- Authors: see the authorship note in `paper/PLAN.md` (Phase 5).
