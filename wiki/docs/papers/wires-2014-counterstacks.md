# Wires et al. 2014 — Characterizing Storage Workloads with Counter Stacks

| Field | Value |
|---|---|
| Authors | Jake Wires, Stephen Ingram, Zachary Drudi, Nicholas J. A. Harvey, Andrew Warfield (Coho Data) |
| Title | Characterizing Storage Workloads with Counter Stacks |
| Venue | 11th USENIX Symposium on Operating Systems Design and Implementation (OSDI '14) |
| Year | 2014 |
| URL fetched | https://www.usenix.org/conference/osdi14/technical-sessions/presentation/wires |
| Verified | 2026-11-10 |
| What I saw | Title, five authors with Coho Data affiliation, session slot "Thursday, August 7, 2014 - 2:15pm", abstract |
| BibTeX key | `wires2014characterizing` |
| Note | The reading list said FAST 2014; the proceedings page places it at OSDI '14. Venue corrected to what the page states. |

## What it did

Introduced the counter stack: instead of storing the stack of cached objects, store
a per-key counter of how many times each key has been requested since a reference
point. Differences of counters reconstruct stack distances, so the structure grows
with the number of *distinct keys referenced in the window* rather than with the
full working set, and it supports MRC construction in sublinear space with
approximate but controlled error. The paper applies it to storage workloads, where
working sets are far too large to materialise.

## The one idea this project borrows

The idea that the MRC is a *derived* object: once you have a compact summary of
the access structure, curves at new capacities are cheap because nothing is
replayed. The hazard model is the extreme version of that position, where the
summary is a feature vector of a few dozen numbers and a fitted survival curve.

## What this project does differently

Counter stacks still index by key, so memory scales with distinct keys in the
window. This project replaces the key index with a model over metadata, which
removes key dependence entirely at the cost of accuracy guarantees. Counter stacks
give a bound; this project gives a point estimate and measures its error.

## Where it lands in this code

`src/mrc/reuse.py` (counter-stack variant used to cross-check the exact stack
algorithm on small shards).
