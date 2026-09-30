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

`main.pdf` is the committed build (tectonic 0.15, 2026-09-30): 6 pages in review mode, no
overfull lines. Rebuild it after any change to `paper/evidence/` or `main.tex`.

A missing evidence set leaves its macros undefined, so LaTeX fails instead of printing a
stale number.

## Before submission

- Check the current GECCO call for the page limit (the draft is 6 pages including
  references), template version, track and anonymity rules. The header uses Kraków,
  Poland, July 12--16, 2027, and WikiCFP lists the deadlines as abstracts 2027-01-19 and
  papers 2027-01-26. These come from WikiCFP and Wikipedia; the official site
  (gecco-2027.sigevo.org) could not be loaded from the build environment, so confirm
  them there.
- The repository link must be anonymised for review (e.g. anonymous.4open.science).
- `refs.bib`, re-checked 2026-09-30:
  - Every entry with a DOI matches Crossref (title, authors, volume, issue, pages, year).
  - JMLR, PMLR, NeurIPS, the NeurIPS Datasets and Benchmarks track, MLSys and arXiv
    entries match the publishers' own pages.
  - Holm (1979) and Kretowski (2019, Studies in Big Data 59) were confirmed by web
    search only, because JSTOR and Springer block automated access.
  - The CART book's ISBN was removed because it could not be verified.
  - The remaining BibTeX warnings are missing publisher/address fields, which ACM's
    style asks for and the proceedings pages do not list.
- Claims attributed to the literature were checked against the sources' text:
  - Barros et al. (2012), Section VII: most comparisons use UCI datasets with greedy
    baselines such as C4.5 and CART, and many report similar accuracy with smaller
    trees. The survey mentions no random-search or pruning-path baseline and gives no
    count of the methods it covers.
  - Freitas (2004): weighted formulas are "to a large extent an ad-hoc approach";
    Pareto and lexicographic approaches are "more principled".
  - Rivera-Lopez et al. (2022): its abstract describes it as "a state-of-the-art
    review".
  - evtree: a penalised single objective, described as "globally optimal".
- Authors: see the authorship note in `paper/PLAN.md` (Phase 5).
