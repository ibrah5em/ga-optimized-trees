# Long-form paper (venue-neutral)

`main.tex` is the full-length version of the study as a standard single-column article
(`article` class, author-year citations), suitable for arXiv or a journal submission. It
covers the same study as `../gecco/`, without the 8-page limit, and adds:

- the pre-registered dataset table and the deviation log;
- the per-dataset K3 table (accuracy, 90% interval, leaves, verdict);
- the one-factor-at-a-time ablation table;
- a threats-to-validity discussion;
- an appendix with the commands that regenerate every result.

Both papers read the same generated numbers, so they cannot disagree.

## Build

```bash
python scripts/paper_assets.py              # writes paper/*/generated/
cd paper/general && latexmk -pdf main.tex   # or: tectonic -X compile main.tex
```

`main.pdf` is the committed build (17 pages). The directory is self-contained, so it can be
uploaded to Overleaf as is, including `generated/`.

## Before submission

- Pick the venue and convert to its template; the text does not depend on the class.
- Authorship and affiliation are taken from the JOSS draft (`../paper.md`); confirm them
  and add other contributors as appropriate (see `../PLAN.md`, Phase 5).
- References without a DOI in `refs.bib` were not machine-checked.
