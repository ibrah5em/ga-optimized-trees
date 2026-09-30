# Provenance

Every result artifact committed to this repository must be attributable to the commit,
config, and seed that produced it. Anything that isn't gets deleted.

Nothing in this directory is tracked except `.gitkeep` placeholders, and `.gitignore`
keeps it that way.

______________________________________________________________________

## The rule

A result artifact may be committed only if all four hold:

1. A command on `main` regenerates it.
1. The config that produced it is committed next to it.
1. The seed that produced it is committed next to it.
1. Its numbers appear nowhere in source code as literals.

If a figure or table can't be regenerated from committed data, it isn't evidence. It's a
picture of a claim.

## Why this exists

Until August 2026 this directory held tables, spreadsheets and figures that no command on
`main` could reproduce. Some came from code that was never merged; the README figures were
drawn from numbers typed straight into a plotting script. The benchmark claims built on them
("46–82% smaller trees at equivalent accuracy") were withdrawn and the files deleted. The
details are in the git history of this file.
