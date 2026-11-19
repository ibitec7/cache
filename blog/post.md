# Guessing the cache miss ratio before you buy the cache

*A hazard model that predicts a miss-ratio curve from request metadata, checked against exact
replay on four production traces, 699 held-out windows.*

---

## 1. The problem is a purchase order

Someone has to decide how many gigabytes the cache tier gets. That decision is made weeks
before the traffic that will fill it exists, and it is expensive in both directions.

Too small, and the miss ratio climbs. Because a cache hit is served in microseconds and a miss
walks to the next tier in milliseconds, the miss ratio is not a gentle cost. Past a knee it is
a latency cliff, and the p99 of the whole service goes with it.

Too large, and you have bought memory that the database or the analytics job needed. Capacity
is not free even when it is cheap, because it is a slice of a fixed budget.

So the question is not "does caching help". It is "what is the miss ratio at each capacity, on
*this* workload, before I commit". The object that answers it is a curve, and this post is
about measuring that curve cheaply and honestly.

## 2. The hierarchy you are shopping in

Nothing here is new, but the sizes matter, because the whole post turns on one level of this
ladder and on the fact that a cache miss is not free.

Every machine already runs a cache hierarchy, and it exists for one reason: fast memory is
small and large memory is slow, by three to five orders of magnitude. Each level is sized and
managed so that the level above it can treat it as though it were both large and fast, and each
level hands its misses down. The numbers below are orders of magnitude, which is all a sizing
argument needs.

| level | typical capacity | typical latency | unit of capacity | who sets the size |
|---|---|---|---|---|
| registers | about 1 KB | 0.3 ns | word | the compiler |
| L1 | 32 to 64 KB | 1 ns | 64 B line | the CPU vendor |
| L2 | 0.5 to 2 MB | 4 ns | line | the CPU vendor |
| L3 / LLC | 16 to 64 MB | 20 ns | line | the CPU vendor |
| DRAM | 8 to 512 GB | 90 ns | line | the memory configuration |
| software cache in memory | 1 GB to 1 TB | 50 to 500 us | **object** | **you, this post** |
| local NVMe | 1 to 100 TB | 20 to 100 us | block | the filesystem |
| object store | effectively unbounded | 20 to 200 ms | object | the account team |

*Table 1. The ladder, with the level this post is about in the middle of it. Latencies are
typical for a current server; the ratio between adjacent levels is what shapes the design, not
the digits. The software cache row is the one whose capacity is chosen by hand, and it is the
only row whose capacity decision is made weeks before the traffic exists.*

Three things differ down that ladder, and they decide what a "capacity" number even means.

**The unit of management.** On-chip caches manage 64-byte lines, the OS manages 4 KiB pages,
and a key-value cache manages *objects*. The size of an object is set by the application and is
not fixed. So a capacity in this post is a count of distinct objects, not a byte count, and the
four traces here average 45 to 122 bytes per object.

**Who owns the decision.** The L1 size is fixed by the CPU vendor. The page cache size is a
sysctl. The capacity of the cache tier in front of a database or an object store is a line item
in a budget, agreed by people who will be asked to justify it later. That is the decision this
post is about.

**What a miss costs.** A miss in DRAM costs about 80 nanoseconds of extra latency, which the
out-of-order engine partly hides. A miss in a *software* cache tier costs a network round trip
to the next tier, which is 100 microseconds to tens of milliseconds, and it is not hidden by
anything. That is why the miss ratio, and not the average latency, is the quantity worth
predicting: the average is dominated by the hits, and the cost is concentrated in the misses.

**Where this post sits.** One tier, in front of a backing store, with these properties: the
working set is a set of objects addressed by key; the request stream is a sequence of key
lookups; the replacement policy is LRU or an approximation of it; and the binding constraint is
capacity, not bandwidth. A request either finds its key (**hit**) or does not (**miss**), and on
a miss the object is fetched from the next tier, returned to the caller, and inserted, evicting
whatever LRU says is coldest.

![path diagram](diagrams/path.svg)

*Figure 1. One request against one tier. The hit path costs microseconds and the miss path costs
milliseconds, and the only knob the operator holds is which keys survive long enough to be the
hit.*

Two consequences follow, and they are the reason the rest of the post is about a distribution
rather than an average.

First, whether a request hits depends on how many *other distinct keys* were touched since the
last time this key was requested. That count is the **reuse distance**, and it is a property of
the request stream, not of the cache. Second, a cache of capacity $C$ that evicts by LRU holds
exactly the keys whose reuse distance is below $C$. So the hit condition reduces to an
inequality on one per-request number, which is what makes the curve estimable at all.

**What this post does not model**, stated up front so no reader has to discover it: admission
policies (TinyLFU, ARC, S3-FIFO and their relatives), prefetching, write-back policies, object
sizes weighted in bytes, and the joint sizing of two tiers at once. Each of those changes the
curve, and a curve predicted under LRU is an approximation under any of them. The contributions
here are the estimator and its evaluation, not a better policy. Section 17 lists the cost of
that assumption again where it matters.

## 3. The miss-ratio curve

Hold the workload fixed and vary the cache size. For every capacity $C$, measure the fraction
of requests that miss. That function is the miss-ratio curve, written $m(C)$.

For a fully associative cache with least-recently-used replacement, $m(C)$ has a clean
definition that needs no simulation. Give each request its **reuse distance** $R$: the number
of *distinct* keys touched since the last time this key was requested. A cache that keeps the
last $C$ distinct keys holds exactly the keys whose reuse distance is below $C$. So:

$$
m(C) = \Pr(R \ge C)
$$

Read it once and it is obvious; read it twice and it is the whole post. The hit condition is
an inequality on a single per-request number, which means the curve is the *tail* of the
distribution of $R$. Predicting a curve is therefore predicting a distribution, not fitting a
line through points.

Two properties follow, and both matter later.

**It is monotone.** More capacity cannot create misses.

**It flattens at a floor $p_c$.** A request whose key has never been seen cannot hit at any
capacity. That floor is the compulsory miss fraction, and no model should try to explain it
away.

![curves diagram](diagrams/curves.svg)

*Figure 2. The measured curves for four production traces, on a log capacity axis. Each trace is
plotted on its own measured grid: 12 capacities for cluster10, 24 for the others, so a line that
stops early has run out of grid, not run out of data. Two readings are worth taking from it. The
steep part of each curve is where capacity buys hit ratio, and the flat part is where it buys
nothing. And cluster10 saturates at a capacity of about five objects, which is why its later
numbers look different from the other three.*

The shape is the point. Between 10 and 10,000 objects the curves move steeply; past a few
hundred thousand they are nearly flat. A cache bought on the flat part is money spent on
nothing, and a cache bought on the steep part is a latency incident waiting for Friday.

| trace | m(1) | p_c floor | capacity at 80% of achievable | p99 reuse distance |
|---|---|---|---|---|
| cluster10 | 0.9353 | 0.5012 | 5 | 8 |
| cluster26 | 0.9417 | 0.0784 | 10,089 | 17,901 |
| cluster45 | 0.8888 | 0.4350 | 78,812 | 1,088,681 |
| cluster52 | 0.9781 | 0.1894 | 42,057 | 493,223 |

*Table 2. Curve shape per trace: the miss ratio at capacity 1, the compulsory floor, and the
capacity needed to reach 80 percent of whatever hit ratio the workload can ever achieve.*

| trace | model: 60 / 80 / 95% of achievable | Kaplan-Meier: 60 / 80 / 95% | disagree by |
|---|---|---|---|
| cluster10 | 3 / 4 / 6 | 2 / 3 / 5 | 0.73x |
| cluster26 | 5,538 / 7,983 / 9,768 | 5,507 / 8,021 / 9,752 | 1.00x |
| cluster45 | 8 / 31,998 / 386,652 | 12 / 15,046 / 300,060 | 0.47x |
| cluster52 | 2,943 / 25,938 / 162,366 | 2,380 / 20,425 / 135,079 | 0.79x |

*Table 3. The same information as a sizing decision, with capacity interpolated in log space
between grid points. For each trace, the capacity needed to reach 60, 80 and 95 percent of the
achievable hit ratio, under the model and under Kaplan-Meier. The last column is Kaplan-Meier's
capacity over the model's: at 0.47 for cluster45, a decision made on the covariate-free curve
buys about twice the memory for the same hit ratio.*

## 4. Why an engineer should care, and what is hard about it

Three uses, in descending order of how often they come up.

**Sizing.** One number per tier, per workload, per quarter.

**Tiering.** If a hot tier and a mid tier split a budget, the split is chosen by comparing two
curves' slopes, not by comparing two averages.

**Regression detection.** A curve moves before a dashboard does. A change in the operation mix
or the key population shows up as a shifted curve while the mean latency is still inside its
noise band.

The hard part is that the honest measurement is a full replay. Exact stack distance needs the
entire trace in memory and one pass per capacity. For a 13.4 billion request trace that is a
real cost, and at design time the trace does not exist yet: you have *today's* workload and a
question about *next month's*.

There is a second cost that is easy to miss. A replay gives the curve for the trace you
replayed. If the traffic shifts, the curve shifts, and the replay has to run again on data that
now exists. A predictor that maps *metadata you already log* to a curve can be fitted once and
re-run per period for the cost of a classification pass, which is the difference between a
quarterly project and a dashboard.

## 5. What has been tried

The literature splits on one axis: what the estimator costs, and what it can promise.

| approach | what it assumes | what it costs | what it cannot do |
|---|---|---|---|
| Exact stack distance (Mattson, 1970) | nothing | one full pass, $O(U)$ memory for $U$ unique keys | run before the trace exists; run per capacity |
| SHARDS | uniform hashing behaves like sampling | many passes with bounded memory | recover exact per-request distances |
| Counter stacks | a small counter array approximates the stack | counter memory per sample | give an exact curve at large capacity |
| Simulation (Talus, libCacheSim) | a cache simulator is faithful | a simulated run per capacity | predict for a policy you did not simulate |
| Kaplan-Meier on reuse distance | the distance distribution is what matters, covariates are not | one pass, no memory | use metadata to separate workloads |
| Learned frequency models | frequency is a proxy for distance | one fit | survive the frequency to distance conversion |
| **This work** | the hazard of reuse is a function of cheap per-request metadata | one fit, then a prediction per window | guarantee in-domain accuracy on an unseen trace |

*Table 4. Where this method sits. The comparison axis is estimator cost against fidelity, not
accuracy alone.*

| | constant memory per request | $O(\text{unique keys})$ memory |
|---|---|---|
| **faithful to the exact curve** | SHARDS (approximate by sampling); this work (measured) | exact stack distance; counter stacks |
| **approximate** | hit-count regression; Kaplan-Meier on reuse distance | (nothing sensible sits here) |

*Table 5. The same comparison without a chart: memory cost against fidelity, one cell per
combination. Only this work's position is backed by a measurement in this repository; the others
are placed from what each method can promise, not from a benchmark run here. The bottom-right cell has no
occupant, which is a small sanity check on the framing: expensive *and* approximate is not a
combination anyone ships on purpose.*

The last row is the contribution and it is deliberately narrow: the survival-function view of
the curve, a discrete hazard with an explicit censoring correction, covariates drawn from
request metadata, and an evaluation that holds out whole windows and whole traces. This is not
the first learned miss-ratio estimator, and the post does not claim that.

## 6. The data

Four traces from the CMU mirror of the Twitter cache traces. A **cluster** here is one
production cache front-end: an anonymized service whose requests were logged in time order and
published as a sorted CSV of `timestamp, key, key_size, value_size, client_id, operation, ttl`.
Fifty-four clusters are published. The object size the cache model uses is `key_size +
value_size`.

Four of them, chosen for contrast:

- **cluster52** is the large one: 13,426,570,607 requests over 6.41 days, 93.09 GB compressed,
  about 24,229 requests per second, mean 48.6 bytes per request.
- **cluster10, cluster26, cluster45** are small shards, 1.0 to 1.7 GB compressed, used for
  development and for the cheap experiments.

Why four and not one: the transfer question needs traces that differ. They do, by orders of
magnitude.

| trace | requests measured | unique keys | cold fraction | R p50 | R p99 | mean object | top operations |
|---|---|---|---|---|---|---|---|
| cluster10 | 5,000,000 | 2,500,051 | 0.500 | 2 | 8 | 1289 B | get 0.50, add/gets 0.50, op1 0.00 |
| cluster26 | 5,000,000 | 394,747 | 0.079 | 1,232 | 17,901 | 1046 B | get 0.71, op2 0.29 |
| cluster45 | 5,000,000 | 2,158,200 | 0.432 | 10 | 1,088,681 | 91 B | get 0.82, add/gets 0.18 |
| cluster52 | 5,000,000 | 932,119 | 0.186 | 1,244 | 493,223 | 252 B | get 0.89, add/gets 0.04, op1 0.03 |

*Table 6. Measured distribution per trace, from the label cache each experiment reads. Cold
fraction is the share of requests whose key was never seen before, which sets the floor $p_c$
in Figure 2. The p99 reuse distance spans roughly four orders of magnitude across these four
traces, so "does the model transfer" is not a small perturbation test.*

One trace needs a warning before its numbers are read. cluster10 has extremely short reuse:
half its requests are first accesses, and 99 percent of the rest are reused within 8 distinct
keys. Its curve is essentially saturated at a capacity of five objects, so it is a useful test of
whether the model stays accurate on a nearly flat curve, and it is not representative of a large
cache. The three other traces span the interesting range.

The operation mix matters more than it looks. Almost nine requests in ten are reads, and `ttl`
is zero for every read and set for every write. A model that treats raw `ttl` as a number sees
a spike at zero that is really the operation column, so the feature is a TTL category plus an
explicit `is_write` flag.

**The unit of experiment is a window.** The trace is cut into windows of 5,000,000 requests in
time order, which is roughly three to four minutes of cluster52 and much longer on the small
shards. Each window is scored separately, so the reported error is a mean over windows and not
a single number over a mixture. The first 60 percent of the windows form the training block,
the next 15 percent are used for early stopping and for the search, and the last 25 percent are
never opened until scoring. That split is fixed once, in code, and every experiment in this
post uses it.

Each window is materialized once into a **label cache**: parquet with the request metadata, the
computed reuse distance, and the event indicator. Every experiment reads that cache rather than
recomputing distances, which is what makes the 24-capacity evaluation of 699 windows a few
minutes of work instead of a day.

## 7. The model: reuse as a lifetime

The curve is a tail probability, so model the distribution directly. Survival analysis is the
branch of statistics built for exactly that, and its vocabulary maps onto this problem without
a metaphor.

**An episode.** Each request opens an episode: the interval until the same key is requested
again. The length of that episode is the reuse distance $R$ of the next access. One episode per
request, millions of episodes per window.

**Censoring.** Some episodes never close inside the trace. The key was requested and then not
seen again before the log ends, so we know it survived past that point and we do not know how
long after. That is **right censoring**, and it is information, not missing data: it says the
reuse distance is *at least* this large.

![episode diagram](diagrams/episode.svg)

*Figure 3. One episode and its two terminal outcomes. A third outcome exists in the data and not
in the figure: the key was never seen before, which is not an episode at all but the compulsory
floor $p_c$. Censoring is what a naive regression has to throw away or invent a label for.*

**Discretization.** Split the distance axis into $K = 32$ log-spaced buckets. Buckets rather
than exact values, because the tail is four orders of magnitude long and a smooth model over
32 bins is easier to regularize than a model over $2^{21}$ distances.

Now the central quantity. For bucket $b$ and feature vector $x$:

$$
h_b(x) = \Pr(d = b \mid d \ge b,\; x)
$$

This is the **discrete hazard**. Read the condition carefully, because it is what makes the
model correct. It asks: given that this episode has *already survived* to bucket $b$, what is
the chance it ends here? Every quantity is conditioned on the state the episode is actually in.

**Survival is a product.** To still be open after bucket $t$, the episode must have failed in
none of the buckets up to $t$:

$$
S(t \mid x) = \prod_{b \le t}\bigl(1 - h_b(x)\bigr)
$$

A product and not a sum, because each bucket's hazard is conditional on having reached it. The
events are chained, so their probabilities multiply. This is the same identity that makes
survival analysis work in medicine, and it is the same reason a Kaplan-Meier curve is a running
product of one-minus-hazard.

**The curve is a read-off.** Put the pieces together:

$$
m(C) = p_c + (1 - p_c)\,\bigl(1 - S(\mathrm{bin}(C-1) \mid x)\bigr)
$$

The second term is the probability that a request could have hit but did not survive to the
capacity bucket. The $p_c$ term is the compulsory floor from Section 3: a first access cannot
hit at any capacity, so it sits outside the survival product rather than inside it. Get that
term wrong and the curve is wrong at every capacity by a constant.

**An arithmetic check, with round numbers.** Suppose one percent of requests are first accesses,
and the hazards for the first three buckets are 0.50, 0.25 and 0.125, all independent of the
features for this example. Then survival is $0.50$, $0.375$ and $0.328$ after buckets one, two
and three, so a capacity in bucket three has miss ratio $0.01 + 0.99 \times (1 - 0.328) =
0.675$. The model's whole job is to get those three hazards right conditional on $x$; the
arithmetic after them is fixed. This is an illustration, not a measurement.

![layers diagram](diagrams/layers.svg)

*Figure 4. The four layers, left to right. Nothing downstream of the hazard is learned: once
$h(b)$ is estimated, the survival product and the curve are arithmetic. That is why the model
can be wrong about a request and still be right about a curve: the errors of millions of
episodes are multiplied into one number per capacity.*

| symbol | meaning | domain |
|---|---|---|
| $R$ | reuse distance: distinct keys touched since the previous access to this key | $\mathbb{N}\cup\{\infty\}$ |
| $d$ | lifetime of one episode, equal to $R$ of its closing access | $\mathbb{N}\cup\{\infty\}$ |
| $b$ | bucket index, 1 to 32, log spaced over the distance range | $1\dots 32$ |
| $h_b(x)$ | discrete hazard: probability the episode ends in bucket $b$, given it reached $b$ | $[0,1]$ |
| $S(t\mid x)$ | survival: probability the episode is still open after bucket $t$ | $[0,1]$ |
| $x$ | 12 metadata features of the request that opened the episode | $\mathbb{R}^{12}$ |
| $p_c$ | compulsory fraction: requests whose key was never seen before | $[0,1]$ |
| $m(C)$ | miss ratio of a fully associative LRU cache of capacity $C$ | $[0,1]$ |
| $C$ | capacity in distinct objects | $\mathbb{N}$ |
| $\delta$ | event indicator, 1 if the reuse was observed, 0 if censored | $\{0,1\}$ |
| $n$ | number of scored windows, the unit of replication | $\mathbb{N}$ |

*Table 7. Every symbol used above, with its domain. There are ten of them and no others are
needed.*

**Where the censoring correction actually lives.** Not in the loss, and not in a post-hoc
adjustment. It lives in how the training rows are built. Each episode becomes a set of rows,
one per bucket it was at risk in. The label is 1 only in the bucket where the reuse happened.
A censored episode contributes rows with label 0 up to its censoring bucket, and no rows after
it. That construction is the **risk set**: the denominator at bucket $b$ is exactly the episodes
still alive at $b$, which is precisely the conditioning in $h_b$.

Treat a censored episode as a non-event for all 32 buckets instead, and you inflate the
denominator with episodes that were never observed that long. The hazard comes out too low,
survival too high, and the curve bends upward at large capacity. The bias is systematic and it
has a sign, which is the worst kind. A regression that simply drops censored episodes has the
opposite problem: it keeps only the episodes that closed quickly, which are the popular keys,
so it over-estimates the chance of a small reuse distance.

## 8. Why a survival model and not a regressor

Three arguments, each with a mechanism rather than a preference.

**1. The estimand is a distribution, not a mean.** A regressor trained to predict $R$ minimizes
error on the mean of $R$. The quantity anyone needs is $\Pr(R \ge C)$ at 24 capacities. The
mean of a heavy-tailed variable with mass at infinity is not a useful summary, and a model that
minimizes squared error on it is optimizing the wrong functional.

**2. Censoring has no honest regression answer.** With a regression target, unobserved episodes
are a hole. Fill it by dropping and you select on the outcome. Fill it by imputation and you
invent data. The hazard formulation consumes them exactly: they contribute "no event" rows and
they leave the risk set at the right time.

**3. The bias and variance land in different places.** This is the part that took measuring.
A per-request hazard is a smooth function of twelve features. A flexible learner on millions of
rows will not overfit the *curve*, because the curve averages over a risk set with millions of
members: individual errors partly cancel, and the survival product smooths further. The
capacity knob is `num_leaves`, and the measured in-domain residual is dominated by a signed
level offset rather than shape noise, with $|\text{bias}|/\mathrm{MAE}$ between 0.74 and 0.97
(Table 14). That is the signature of a low-variance, slightly miscalibrated fit, which is the
good failure mode: a scalar could remove most of it, and a scalar is cheap.

So the trade is explicit and it is made in the right place. Take a high-variance learner,
accept that it can be miscalibrated in level, and make the *distribution* the object that gets
averaged. Do not ask one small tree ensemble to be sharp and unbiased at the same time.

## 9. The learner

LightGBM. Concretely: gradient-boosted decision trees over histogram-binned features.

Define the pieces, because the acronyms carry no information.

**Gradient boosting.** Fit a small tree on the errors of the current model, add it with a
shrinkage factor, repeat. Each tree is fitted to the gradient of the loss, which for a binary
logistic loss is the residual between the label and the predicted probability.

**Histogram binning.** Each feature is bucketed once into 255 bins. A split search then costs
$O(\text{bins})$ per feature instead of $O(\text{rows})$. On a design matrix with 20 million
rows and 12 features this is the difference between seconds and minutes per tree.

**Leaf-wise growth.** Trees grow by splitting whichever leaf reduces the loss most, not level
by level. `num_leaves` is therefore the real capacity knob, and 63 leaves is a reasonable
ceiling for this data.

**GOSS** is an optional sampling scheme in LightGBM: keep the rows with large gradients and
subsample the rest. **This project does not use it.** It is worth saying why we did not try
harder: down-sampling the small-gradient rows changes the effective label distribution that
$S(t)$ is read from, and the risk-set construction is the thing that makes the hazard
unbiased. Trading that away for throughput on a fit that already takes a minute is a bad trade.

**Boosting mode.** Plain `gbdt`, no Dart, no random forest mode.

| parameter | default | tuned range | what it controls |
|---|---|---|---|
| `objective` | `binary` | frozen | binary logloss, one row per (episode, bucket) |
| `metric` | `binary_logloss` | frozen | early stopping on validation logloss |
| `num_leaves` | `63` | 20 to 63 | capacity of one tree, the main regulariser here |
| `learning_rate` | `0.05` | 0.05 to 0.11 | shrinkage per tree |
| `min_data_in_leaf` | `100` | 50 to 100 | minimum rows per leaf, guards rare buckets |
| `feature_fraction` | `0.9` | frozen | column subsample per tree |
| `bagging_fraction` | `0.8` | frozen | row subsample per iteration |
| `bagging_freq` | `1` | frozen | bagging every iteration |
| `lambda_l2` | `1.0` | 0.003 to 1.0 | L2 penalty on leaf values |
| `deterministic` | `True` | frozen | replicate runs are bit-comparable |
| `boosting` | `gbdt` | not tuned | GOSS is available and **not used** |

*Table 8. Shipped parameters, the ranges the search explored, and what each one controls. The
defaults are what the retrain curve used, where the search was frozen on purpose.*

Two choices deserve a line each. `deterministic=True` with `force_row_wise=True` makes replicate
runs bit-comparable at a small throughput cost, which is what lets a result be re-run and
compared instead of trended. And `min_data_in_leaf=100` matters more than it looks: the last
few buckets are sparse, and a leaf with three rows in a rare bucket produces a hazard estimate
with enormous variance that the survival product then compounds.

## 10. Loss, validation and class balance

**It is a classifier, not a regressor.** The label is binary: did this episode end in this
bucket (1) or not (0). The loss is binary logloss on the expanded table of (episode, bucket)
rows. Early stopping uses validation logloss over 200 to 400 rounds.

**No class weighting is applied.** No `scale_pos_weight`, no `is_unbalance`. The label is rare
per row, since each episode contributes one event row to 32 rows of "not yet", but re-weighting
would be wrong here. Predicted probability *is* the quantity the curve integrates: $S(t)$ is a
product of $1 - h_b$, so multiplying the event class to balance the rows would shift every
hazard upward and bend the curve at every capacity. The imbalance is a property of the expanded
table, not a defect of the sample.

One caveat stays on the record. The loss averages over *rows*, and rows are not independent:
32 rows share one episode. So the loss's implicit weighting is not the deployment weighting,
which is per request. We did not correct for it, and the measured level offset (Section 14) is
consistent with the model being slightly conservative, but that link is an inference and not a
measured mechanism.

**Validation is a block, not a fold.** The split is time-ordered: train, then validation, then
test, each contiguous. Shuffling requests across time would leak the future into the fit, and
the entire point is to predict a period that did not exist at fit time.

## 11. Hyper-parameter search

Optuna, with the tree-structured Parzen estimator (TPE) sampler.

Why not grid search: a grid costs the product of the axes. With six axes at five values each
that is 15,625 fits, and it spends most of them in regions that are obviously bad. Why not
random search: it is a fine baseline, it does not waste runs on a lattice, and in high
dimensions it beats a grid. But it learns nothing from the trials it has already run.

TPE models the objective's distribution separately for good and bad trials and samples where
the expected improvement is high. It uses its history, which is what a search should do.

**And then a measurement that humbled the method.** At 12 trials, cluster10, cluster26 and
cluster45 returned *identical* chosen hyper-parameters, and the Spearman rank correlation
between their trial-value rankings was exactly 1.000. That is not three traces agreeing. Seeded
TPE spends its first 10 trials on random draws, so at 12 trials the "search" was two informed
trials dressed as a ranking, and what the ranking actually captured was capacity order across
traces.

Continuing the same studies past trial 20 made them diverge per trace (`num_leaves` 24, 20, 25;
learning rate 0.110, 0.104, 0.103) and improved every test MAE: cluster10 from 0.003704 to
0.003637, cluster26 from 0.001762 to 0.001518, cluster45 from 0.001517 to 0.001313. The studies
reported here ran 35, 29, 30 and 20 trials.

The rule we now apply: **a study is reported only if it ran past trial 20.** A study stopped at
10 or 12 trials is random search wearing a sampler's name. That is a general lesson, not a
detail of this project.

| trace | trials in the study | best validation MAE | test MAE after refit |
|---|---|---|---|
| cluster10 | 35 | 0.003085 | 0.003637 |
| cluster26 | 29 | 0.000610 | 0.001518 |
| cluster45 | 30 | 0.001490 | 0.001313 |
| cluster52 | 20 | 0.001228 | 0.003834 |

*Table 9. Trial counts from the study databases, the best validation-block error each study
found, and the test error after refitting the chosen parameters on the full training block.
The two error columns are different blocks and are not expected to agree: the study optimizes
the validation block, and the last column is the only number that is out of time. cluster52's
study is shorter because its windows are expensive.*

## 12. How it runs

The pipeline is deliberately shaped like the deployment question.

![pipeline diagram](diagrams/pipeline.svg)

*Figure 5. Read a window, pool the training block, fit once, freeze, then score each test window
independently. The freeze is the contract: no test window is opened before it.*

**Streaming, not sliding, and not expanding.** Windows of 5,000,000 requests are read in time
order from parquet and discarded after their histogram and a sample of rows are taken. Peak
memory is one window plus the pooled training rows, not the trace.

**One frozen fit.** The model is fitted once on a uniform sample of the first 60 percent of the
trace. It is never refit per window, and it is not an expanding-window or online model: the
weights do not move after the fit. This makes the reported numbers a deployment claim rather
than an optimistic one, because nothing in the scoring step has seen the future.

**Two details that keep the claim honest.** The bin grid is derived from the *train block only*,
because at deployment the later periods do not exist. And every score file is invalidated when
it is older than the model file, because a stale score reported under new parameters is a
fabrication with a timestamp.

Cost, measured on this machine: one 5M-request window takes about 12 seconds to read and label,
and a fit takes 30 to 90 seconds depending on the row budget. Scoring is the parallel part, and
it is embarrassingly parallel because each window writes its own file.

## 13. Two validations, two questions

![validate diagram](diagrams/validate.svg)

*Figure 6. Out-of-time keeps the trace and moves the clock. Leave-one-cluster-out keeps the
clock and moves the workload. Both are reported, because either one alone flatters a different
kind of model.*

**Out-of-time (OOT).** Fit on the first 60 percent, score every window of the last 25 percent.
The trace, the capacity grid and the feature distribution are fixed; only time moves. This
answers the deployment question for a cache you already operate: does this week's model describe
next week.

**Leave-one-cluster-out (LOCO).** Fit on three traces, score every window of the fourth. The
code is fixed and the distribution moves. This answers the new-customer question: can a model
trained on workloads we have seen work on one we have not.

The two are reported together because each alone is misleading. OOT alone says nothing about a
workload you have never measured, and LOCO alone cannot distinguish a model that fails on drift
from one that fails on a different key population.

## 14. Results

### Out of time

| trace | n windows | MAE | median | Kaplan-Meier MAE | ratio | MRE | windows won |
|---|---|---|---|---|---|---|---|
| cluster10 | 7 | 0.003637 | 0.003273 | 0.027656 | 0.132 | 0.0061 | 7/7 |
| cluster26 | 8 | 0.001518 | 0.001794 | 0.005109 | 0.297 | 0.0038 | 8/8 |
| cluster45 | 12 | 0.001313 | 0.000977 | 0.024877 | 0.053 | 0.0021 | 12/12 |
| cluster52 | 672 | 0.003834 | 0.003184 | 0.030559 | 0.125 | 0.0070 | 672/672 |

*Table 10. Every window of the last 25 percent of each trace. MAE is the mean over the 24-point
capacity grid of the absolute difference between predicted and exact miss ratio. The ratio
column divides that by the Kaplan-Meier error on the same windows. `n` is the number of windows,
and it is not decoration: a mean over 7 windows and a mean over 672 are different evidence.*

The model beats the covariate-free floor on **all 699 scored windows** across the four traces,
with no exception. Three of the four are accurate to between 2 and 7 parts in a thousand of miss
ratio.

MAE is the right metric here and it is worth one sentence, because a relative error would
flatter the model. Near the compulsory floor every predictor is nearly right in ratio terms,
including the bad ones, so a relative metric is dominated by where the curve is flat and easy.
An absolute error in miss ratio is also the unit the decision is made in: one part in a thousand
of miss ratio is a latency change of microseconds on hits and a fraction of a percent of traffic
reaching the next tier.

cluster52 deserves its own sentence, because it is the only row whose test block is a
distribution rather than a handful of windows: 672 of 672 windows, 94 shards, no subsampling.
Its block-bootstrap 95 percent interval on the mean MAE is **[0.003416, 0.004318]**, and its
per-window interquartile range is [0.002122, 0.005872]. The bootstrap resamples contiguous
blocks, not single windows, because neighbouring windows share a working set and an i.i.d.
bootstrap would report an interval that is too narrow.

| baseline | status | c10 | c26 | c45 | c52 | reading |
|---|---|---|---|---|---|---|
| hazard GBDT (this work) | ran on all four | 0.0077 | 0.0055 | 0.0003 | 0.0003 | the model |
| Kaplan-Meier | ran on all four | 0.0439 | 0.0118 | 0.0200 | 0.0248 | floor with no covariates |
| hit-count regression | ran on all four | 0.0793 | 0.0793 | 0.1097 | 0.1134 | worse than the floor |
| single-row (oracle) | ran on all four | 0.0000 | 0.0000 | 0.0000 | 0.0000 | reads the truth, not a competitor |

*Table 11. The four things that were scored on the same windows, on the same 24-point grid. The
hazard model is the only learned one; Kaplan-Meier is the same estimator with no covariates, and
it is the floor the model has to beat; the hit-count regression is the obvious alternative and it
loses to the floor; `single_row` reads the answer and is an oracle, reported so the gap between
"fits the curve" and "knows the curve" is visible.*

![ratio diagram](diagrams/ratio.svg)

*Figure 7. The same eight ratios as bars: model error divided by Kaplan-Meier error, one bar per
trace and protocol. A bar of 1.0 would mean the covariates bought nothing; every bar here is far
below it, and the worst case is a third of the floor. The out-of-time bars are the ones that
answer the deployment question; the leave-one-out bars answer the transfer question, and they
are lower for three of the four traces, which is the opposite of what the absolute errors say.*

### Out of distribution

| held out | n windows | MAE | Kaplan-Meier MAE | ratio | MRE | MAE vs own-data model |
|---|---|---|---|---|---|---|
| cluster10 | 7 | 0.002756 | 0.160934 | 0.017 | 0.0047 | 0.76x |
| cluster26 | 8 | 0.006770 | 0.156697 | 0.043 | 0.0378 | 4.46x |
| cluster45 | 12 | 0.005445 | 0.097702 | 0.056 | 0.0097 | 4.15x |
| cluster52 | 667 | 0.016266 | 0.144415 | 0.113 | 0.0324 | 4.24x |

*Table 12. Trained on three traces, scored on the fourth. The last column divides the held-out
error by the same trace's own-data error from Table 10.*

Four readings, in order of how much they cost to learn.

**The functional form transfers.** Every fold beats Kaplan-Meier by between 9 and 60 times. On a
trace the model has never seen, it is still a usable predictor of the curve's shape. Nothing
collapses.

**The absolute level does not.** Three of four folds lose about a factor of four in MAE, and the
held-out cluster52 error is 1.63 percentage points of miss ratio, which is enough to mis-size a
cache by a factor. A model trained elsewhere is a good prior and a poor final answer.

**The loss is distributional, not starvation.** The LOCO models were trained on 180 million
requests each, against 83.5 to 120 million for the own-data models. More data produced worse
error on three of four traces. That is a clean argument that the gap is the distribution
shifting, not the model needing more rows.

**One trace inverts, and the retrain curve explains it.** cluster10's held-out model
(0.002756) beats its own-data model (0.003637). That looked like a bug until the retrain curve
(Figure 8) showed cluster10 is still falling steeply when more of its own history is pooled: its
binding constraint is coverage of its own periods, so a broader set of other workloads is a
better prior than the narrow slice it was fitted on. Two independent experiments, one
mechanism.

### How much history a new trace needs

| trace | training windows | requests in the fit | mean test MAE | n test windows |
|---|---|---|---|---|
| cluster10 | 1 | 2,499,949 | 0.017567 | 7 |
| cluster10 | 5 | 3,636,360 | 0.006046 | 7 |
| cluster10 | 12 | 3,636,360 | 0.004259 | 7 |
| cluster26 | 1 | 1,107,137 | 0.003057 | 8 |
| cluster26 | 5 | 1,034,307 | 0.001861 | 8 |
| cluster26 | 12 | 1,010,840 | 0.001706 | 8 |
| cluster45 | 1 | 1,902,911 | 0.001398 | 12 |
| cluster45 | 5 | 1,863,994 | 0.001313 | 12 |
| cluster45 | 12 | 1,873,488 | 0.001333 | 12 |

*Table 13. Hyper-parameters frozen with `--trials 0`, the same test windows at every budget, and
the fit row count held fixed at about 3.6 million rows. The axis that moves is therefore the
number of distinct 5 million request windows pooled into the training sample, not the number of
rows fitted.*

That distinction is worth a sentence, because it is easy to get wrong and this experiment got it
wrong first. The pooled sample is capped by a row budget that is divided across the pooled
windows, so adding windows buys *coverage*, not volume. What the curve measures is how much
history the model needs, at constant fit cost. The earlier description of these runs as 5, 25
and 60 million training rows was wrong; the artifact records 2.5 to 3.6 million rows for every
point.

![retrain diagram](diagrams/retrain.svg)

*Figure 8. Pooling 12 windows instead of 1 drops cluster10's error by 4.1 times at the same row
count, so the gain is coverage and not volume. cluster26 gains 1.8 times and flattens. The y
axis is the mean test MAE, so 0.0043 is 4.3 parts in a thousand of miss ratio.*

cluster10 is still falling at 12 windows, which is the fact that explains the inversion in
Table 12. cluster45's line is flat, and its three points sit inside its own per-window spread of
0.0007 to 0.0035. That line is a **null result**: this experiment cannot say what cluster45's
history demand is, and it is reported as a null rather than as evidence of saturation.

### What the residual looks like

| trace | MAE | mean bias | abs(bias)/MAE | mean worst grid point | worst / MAE |
|---|---|---|---|---|---|
| cluster10 | 0.003637 | -0.003365 | 0.93 | 0.010231 | 2.8x |
| cluster26 | 0.001518 | -0.001338 | 0.88 | 0.004436 | 2.9x |
| cluster45 | 0.001313 | -0.000976 | 0.74 | 0.004577 | 3.5x |
| cluster52 | 0.003834 | -0.003728 | 0.97 | 0.009771 | 2.5x |

*Table 14. In-domain residual decomposition. Bias is the mean signed error of the curve; a
negative bias means the model under-predicts the miss ratio.*

This is the finding that was not planned. In-domain, the residual is almost entirely a
**constant level offset**: $|\text{bias}|/\mathrm{MAE}$ is 0.74 to 0.97, the sign is negative
every time, and the worst single grid point is only 2.5 to 3.5 times the MAE. The model
under-predicts the miss ratio by 0.1 to 0.4 percentage points, uniformly across capacities.

That matters for two reasons. Practically, the failure direction is under-provisioning, which
is the expensive one. Statistically, a uniform offset is a calibration error and not a shape
error, so a per-trace scalar multiplier would remove most of it. That recalibration is a
hypothesis: it has not been run, and it is the first item on the list of what to do next.

Out of domain the picture changes: for cluster10 and cluster26 the worst grid point is 9 to 10
times the MAE, so the residual is no longer a clean offset. Level calibration is an in-domain
tool, not a general fix.

## 15. Can we train once and keep it deployed

The question behind the question, and the answer has three parts.

**For a cache you already run: yes, with eyes open.** Out-of-time error is 0.2 to 0.7 percent
of miss ratio on three traces and holds on every window. The model needs no retraining to stay
useful across the measured horizon, which is the last 25 percent of each trace.

**For a cache you have never measured: no, not on its own.** Beats the floor on every window,
loses about a factor of four in absolute accuracy. Use it to plan the first week, then collect a
few tens of millions of requests, refit, and re-tune.

**Drift will show up as bias first.** Since the in-domain residual is a level offset, a
monitoring rule that watches the mean signed error of the curve is more sensitive than one
watching the shape. That is a design implication of Table 14 and it is testable, which is more
than a generic "monitor for drift" sentence.

## 16. What we got wrong

Three retractions. They are here because a results section without them is a sales document.

| claim we made | what was wrong | corrected |
|---|---|---|
| 12-trial Optuna studies had converged | seeded TPE spends its first 10 trials on random draws, so three traces returned identical parameters and a rank correlation of 1.000 | continued to 26 to 34 trials; every test MAE improved |
| three traces re-tuned better after a refit | the scorer reused window files older than the model, so the same scores were reported under new parameters, identical to 16 digits | rescoring is keyed on the model's mtime |
| `single_row` scored a perfect curve | it reads `eval_rtilde` for the block it scores, so its own bin decides the hit | it is an oracle, reported as one, never as a competitor |

*Table 15. Claims that were wrong, the mechanism that made them wrong, and what replaced them.*

The pattern across all three is the same: the failure was in the measurement harness, not in the
model. A study that stops before its sampler starts, a cache that validates on the wrong file's
timestamp, a baseline that reads the answer. Each one produces a plausible number, which is
exactly why each one had to be caught by an experiment rather than by review.

A fourth correction belongs here too, because it is the same kind of error and it was caught the
same way: the retrain experiment was described as pooling 5, 25 and 60 million training rows,
and the artifacts record 2.5 to 3.6 million for all three points. The label was wrong, the
measurement was not.

## 17. Limitations

**LRU only.** The curve is defined by LRU hits. A real cache runs an eviction policy with
admission and a probabilistic eviction (S3-FIFO, TinyLFU, and friends), whose miss ratio is not
$m(C)$. Using this curve for that policy is an approximation with no measured error bound here.

**Object capacity, not bytes.** The code can do a byte-weighted curve, and the four traces here
are reported in objects. A byte curve is what a provisioning decision actually wants.

**One corpus.** Four traces from one mirror of one service. The pattern across them is
informative, and it is not evidence about storage, CDN or database workloads.

**A fixed censoring convention.** Episodes are censored at the trace boundary. A streaming
deployment censors at the point of prediction, which is a different (and shorter) horizon.

**32 buckets.** The top bucket merges every distance beyond the cap. The curve is therefore
exact only below that cap, and the cap is set from the train block's p99.9.

**No online adaptation.** The model is frozen after the fit. Section 15 recommends retraining,
which is a schedule, not a method.

**What we would do next**, in the order the evidence supports it. One, the per-trace scalar
recalibration that Table 14 implies, measured rather than asserted: it is a one-parameter fit on
the validation block and should remove most of the level offset. Two, a byte-weighted curve, so
the output is in the unit a provisioning decision uses. Three, an admission-aware extension:
compare LRU curves against the same trace run through a TinyLFU or S3-FIFO simulator, and report
the gap rather than assuming it is small. Four, a second corpus from a different storage stack,
since the transfer result is currently measured on one family of workloads.

## 18. Conclusion

The curve is a tail probability, so model it as one. A discrete-time hazard over the reuse
distance, with covariates from request metadata and censoring handled in the risk-set
construction, predicts the miss-ratio curve on every one of 699 held-out windows and beats the
covariate-free floor on all of them. Trained on other workloads it still beats the floor, but
loses about a factor of four in absolute accuracy, which is the honest boundary of the claim:
the method transfers, one set of weights does not.

The two findings worth carrying forward are both about where the error lives. In domain it is a
level offset, not shape noise, which makes a scalar recalibration the obvious next experiment.
And the search was a startup artefact until it ran past trial 20, which is a rule about
reporting rather than a fact about Optuna.

| step | command | produces |
|---|---|---|
| fetch a trace | `bash scripts/fetch_traces.sh cluster45` | `data/raw/traces/` |
| build the shard | `env -u PYTHONPATH uv run python -m mrc.cli build cluster45` | `data/publish/cluster45/` |
| fit and score one window | `env -u PYTHONPATH uv run python -m mrc.cli train cluster45` | `artifacts/tables/cluster45_metrics.json` |
| tune | `env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 30` | `artifacts/study_cluster45.db` |
| out of time, all windows | `bash scripts/farm_oot.sh` | `artifacts/oot/cluster45/` |
| leave one out | `bash scripts/farm_loco.sh` | `artifacts/loco/cluster45/` |
| render the figures | `env -u PYTHONPATH uv run python scripts/blog_diagrams.py` | `blog/diagrams/*.svg` |
| rebuild this post | `env -u PYTHONPATH uv run python scripts/blog_build.py` | `blog/post.md` |

*Table 16. Reproduction, in order. Every number in this post comes out of the artifacts these
commands write; the post itself is generated by the last one.*

---

### Notes on this post

The five schematics are Mermaid source in `blog/diagrams/src/`, rendered to SVG by
`scripts/blog_diagrams.py`. Each rendered file carries both palettes and switches on
`prefers-color-scheme`, so the figures are correct on a white page and on a dark one without a
second asset. The three data figures are drawn from the artifacts by `scripts/blog_figures.py`
on a real log capacity axis, with a mid-tone palette that survives either background. Every
table is generated from the same artifacts by `scripts/blog_build.py`: a number in this post
that no artifact supports cannot appear, because it would have to be typed by hand and the build
does not type numbers. The companion wiki pages under `wiki/docs/` carry the full tables, the
per-window files and the retracted values.
