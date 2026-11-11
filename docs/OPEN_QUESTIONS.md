# Open questions

The three questions from the master plan section 6. None of them blocks Phases A to G. Each has a
default already in force (recorded in `logs/DECISIONS.md`), so a worker never waits: it proceeds on the
default and the answer either confirms it or forces a rewrite of a named artefact.

Format per question: what is being decided, when it must be decided by, the options, the default, what
that default costs, and who decides.

---

## Q1. Byte-capacity curve: headline or appendix?

**Decision needed by:** before H1 starts (the first figure commit). Phase H, task H1.

**Options**

| Option | What changes |
|---|---|
| A. Object capacity is primary, byte capacity in the appendix | one primary figure (`plot_mrc` on object capacities), one appendix figure (`mrc_curve(labels["B"], caps, own_size=labels["obj_size"])`), Methods keeps the byte paragraph short |
| B. Both are headline | two figures in the body, the capacity axis gains a unit column in every table, and the appendix shrinks; the byte curve must be labelled approximate wherever it appears |

**Default if unanswered:** A.

**Consequence of the default:** the byte curve is reported once, in the appendix, with the measured gap
against the object curve and a one-line statement of why it is not exact (objects larger than capacity
break the stack-distance argument). No table in the body carries a byte column. If the answer is B, H1's
figure list and H3's table columns both change, and the body page budget is 8 to 10 pages measured by
`G-H4-report`.

**Evidence behind the default:** the frozen contract in master section 2 names Mattson stack distance
"exact for object capacities" only; `docs/RISKS.md` R2 risk row and the `logs/WORKLOG.md` entry
"byte stack distance approximate (6.5e-5 on real data)" state the same asymmetry.

**Decider:** the user (course supervisor for the report's framing).

---

## Q2. Is the cluster52 20 M-request run required?

**Decision needed by:** before the cluster52 run is dispatched (Phase F, task F4, or later as a
standalone `mrc train cluster52`). Not before A to E.

**Options**

| Option | What changes |
|---|---|
| A. 5 M window on four clusters is enough for the robustness claim | the robustness section claims window sensitivity only, via the four-cluster comparison; cluster52 is not run at scale |
| B. A 20 M-scale run is required | the run is a 10 M-request run, not 20 M, because 20 M x 3.73 = 74,600,000 grouped rows breaches `MAX_GROUPED_ROWS = 60,000,000`; a 20 M run needs a row guard change, which is a frozen-constant edit |

**Default if unanswered:** A, with the 10 M cluster52 run kept as an optional sensitivity appendix.

**Consequence of the default:** no claim about behaviour beyond a 5 M window. The report states the
window length in every caption (frozen contract: "report the window length in every caption"). If B is
required, the cost decision and `MAX_GROUPED_ROWS` change must be logged in `logs/DECISIONS.md` and
`logs/WORKLOG.md` before the run, and every artefact produced under the old guard becomes stale.

**Evidence behind the default:** cost projection in `logs/DECISIONS.md` (5 M -> 18,650,000 rows, 20 M ->
74,600,000 rows, 10 M -> 37,300,000 rows), and master section 6 listing the scale run as a "sensitivity
check" rather than a deliverable.

**Decider:** the user.

---

## Q3. Does the report lead with accuracy or with interpretability?

**Decision needed by:** before H4 starts (the report-fill task). It fixes figure order and the first
sentence of the abstract.

**Options**

| Option | What changes |
|---|---|
| A. Accuracy first | H1's MRC figure is figure 1; the abstract opens with the MAE result; interpretability is the second contribution, placed after the evaluation section; the G figures follow H1 |
| B. Interpretability first | the permutation/SHAP figure moves to figure 1; the abstract opens with which features drive reuse; the MRC figure moves to the evaluation section; H1 is no longer the first commit of Phase H |

**Default if unanswered:** A.

**Consequence of the default:** the abstract's first sentence is the accuracy claim, and the report's
argument is "the estimator matches the ground truth, and here is what it learned". If the answer is B,
`G-H1-mrc` still gates H4 but H4's section order and the slides' `06-results.tex` order change, and the
word budget shifts from the evaluation section to the interpretability section.

**Evidence behind the default:** the frozen primary metric is accuracy MAE (master section 2, row
"Primary metric"; `logs/DECISIONS.md` first line of the decision log), and every numeric acceptance
threshold in master section 5 is an accuracy threshold.

**Decider:** the user.

---

## How an answer is applied

1. The user answers in the card thread or in chat.
2. `mrc-plan` appends one line to `logs/DECISIONS.md` in the form
   `UTC date | question answered: <option> | <the answer verbatim or its source> | yes/no`.
3. If the answer contradicts the default, the affected phase plan is edited in the same session and the
   edit is listed in the answer's decision line.
4. No worker is re-dispatched on the strength of an answer alone; the affected card is re-opened, which
   is the only thing that restarts work.
