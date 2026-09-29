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
- Verify every entry in `refs.bib` against the publisher's record; they were written
  from memory of the literature, not exported from a database.
- Authors: see the authorship note in `paper/PLAN.md` (Phase 5).
