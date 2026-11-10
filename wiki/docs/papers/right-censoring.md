# Right-censoring

## What it is

An episode is right-censored when the trace ends (or the observation window closes)
before the key is requested again. We then know only that the reuse distance
exceeds the remaining window, not what it is. Censoring is a property of the
observation process, not of the cache, and it is the single most common way to get
a reuse-distance distribution wrong.

## Why it is needed here

Every trace is a finite window. The last request for each key inside the window has
no observed successor, so a naive estimator sees only the keys that happened to be
reused quickly. That subpopulation is precisely the cache-friendly one, so dropping
censored episodes biases the estimated reuse distance downward, which biases the
MRC downward at every capacity, which overstates the benefit of adding memory. The
Twitter corpus makes this worse than usual because TTL expiry removes objects
without a later request, so "no successor in the window" has two causes.

Sources for the correction: Kaplan & Meier 1958 ([page](kaplan-meier-1958.md)) for
the product-limit estimator, Cox 1972 ([page](cox-1972.md)) for the covariate form,
Allison 1982 (Discrete-Time Methods for the Analysis of Event Histories,
Sociological Methodology 13, 61, DOI 10.2307/270718, verified via Crossref
2026-11-10) for the discrete-time treatment.

## Formal statement

For episode $i$, let $d_i$ be the true lifetime and $\omega_i$ the time the window
closes. We observe

$$
\tilde d_i = \min(d_i, \omega_i), \qquad \delta_i = \mathbf{1}\{d_i \le \omega_i\}.
$$

Under independent censoring, the likelihood of the observed data is

$$
L = \prod_{i:\,\delta_i=1} f(\tilde d_i \mid x_i) \prod_{i:\,\delta_i=0} S(\tilde d_i \mid x_i),
$$

and the nonparametric maximiser is the product-limit estimator
$S(t) = \prod_{s\le t}(1 - d_s/n_s)$ with $n_s$ the number at risk.

| Symbol | Meaning | Domain |
|---|---|---|
| $d_i$ | true lifetime of episode $i$ | $\mathbb{N}\cup\{\infty\}$ |
| $\omega_i$ | censoring time, window remaining | $\mathbb{N}$ |
| $\tilde d_i$ | observed lifetime, $\min(d_i,\omega_i)$ | $\mathbb{N}$ |
| $\delta_i$ | event indicator, 1 observed, 0 censored | $\{0,1\}$ |
| $f$ | density of the lifetime | $\ge 0$ |
| $S$ | survival function | $[0,1]$ |
| $n_s$ | number at risk at bucket $s$ | $\mathbb{N}$ |
| $d_s$ | events in bucket $s$ | $\mathbb{N}$ |

Assumption to state out loud: **independent censoring** means the window closes
independently of the lifetime. This is violated when TTL is correlated with reuse
(a short TTL both prevents reuse and shortens observation). Reported curves are
therefore also computed with TTL-expired episodes treated as events, and the gap
between the two versions is reported rather than hidden.

## Worked micro-example

Trace A B A C B D again, $K=4$, four censored episodes. Three ways to handle them,
all evaluated at the same two capacities:

| Handling | $S(1)$ | $S(2)$ | $m(2)$ | Error vs $m^\ast(2)=0.8333$ |
|---|---|---|---|---|
| Correct: censor at $K=4$ (Kaplan-Meier) | 0.8333 | 0.6667 | 0.8333 | 0.0000 |
| Drop censored episodes | 0.5000 | 0.0000 | 0.5000 | $-$0.3333 |
| Treat censoring as an event at 4 | 0.8333 | 0.6667 | 0.8333 | 0.0000 at $C=2$, but $m(5)=0$ |

Reading the arithmetic of the bad row: dropping leaves two episodes, so
$S(1) = 1 - 1/2 = 0.5$ and $S(2) = 0.5 \times (1 - 1/1) = 0$. The estimator claims
half the requests miss at capacity 2 when five of six actually do. The error is
$-0.3333$ absolute, which on a real trace is the difference between "buy 2x
capacity" and "buy nothing".

The third row shows the opposite failure. Treating censoring as an event is
harmless where no event occurs in that bucket, but it forces $S(4)=0$ at
$n_4=4$, $d_4=4$, so the model asserts $m(C)=0$ for $C \ge 5$: every key reused
within distance 4. On this toy trace that happens to be true, which is exactly why
the failure is dangerous, it is invisible on small examples and wrong on
week-long ones.

So the estimator must **keep censored episodes in the risk set up to their
censoring bucket, then remove them without counting an event**. That is one line of
code and the difference between a usable curve and a plausible-looking wrong one.

## Where this lives in the code

`src/mrc/hazard.py` (`censor_time`, risk-set updates, the TTL-expiry variant),
`src/mrc/features.py` (`delta` and `omega` as columns, never as features).
