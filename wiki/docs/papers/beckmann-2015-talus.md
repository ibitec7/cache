# Beckmann & Sanchez 2015 — Talus

| Field | Value |
|---|---|
| Authors | Nathan Beckmann, Daniel Sanchez |
| Title | Talus: A Simple Way to Remove Cliffs in Cache Performance |
| Venue | 2015 IEEE 21st International Symposium on High Performance Computer Architecture (HPCA), pp. 64–75 |
| Year | 2015 |
| DOI | 10.1109/HPCA.2015.7056022 |
| URL fetched | https://api.crossref.org/works/10.1109/hpca.2015.7056022 |
| Verified | 2026-11-10 |
| What I saw | Title, both authors, `issued` 2015-02, container "2015 IEEE 21st International Symposium on High Performance Computer Architecture", pages 64–75 |
| BibTeX key | `beckmann2015talus` |
| Note | The reading list said ISCA 2015; the registered venue is HPCA 2015. Corrected to the registered venue. |

## What it did

Observed that MRCs of real workloads have cliffs: capacities at which the miss
ratio drops sharply because an entire popular working set suddenly fits. Around a
cliff, competition between two applications for the shared cache is pathological,
and the paper proposes Talus, which uses partial MRCs to smooth demand and remove
those cliffs at the shared-cache level. The mechanism is a combination of
per-application MRC estimation and bandwidth-style allocation of capacity.

## The one idea this project borrows

The cliffs are the reason MRC estimation must be evaluated *per capacity* rather
than on average. A model that smooths a cliff into a gentle slope can have a low
MAPE and still be useless for capacity planning, because the operating point sits
on the cliff. This is why [Results](../results.md) uses MaxErr as an acceptance
criterion and why the toy trace is checked capacity by capacity.

## What this project does differently

Talus estimates partial MRCs by running the stack algorithm on a sample of the
access stream and then allocates capacity. This project does not allocate
anything: it predicts the whole curve and treats curve error, especially at
cliffs, as the deliverable.

## Where it lands in this code

`src/mrc/plot.py` (residual plots that make cliffs visible), the MaxErr criterion
in `src/mrc/metrics.py`.
