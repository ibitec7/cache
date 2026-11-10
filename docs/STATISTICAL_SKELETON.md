# Statistical Skeleton: Hit-Ratio Curves from Reuse Distance

**Status:** specification, frozen before modelling. Any later deviation is recorded as a deviation, not silently applied.
**Scope:** the statistical object of the project, its identification assumptions, its estimator, and its validation. Implementation detail lives in `src/`; this document is the contract that `src/` must satisfy.
**Data:** request-level traces from an in-memory cache cluster (Twitter, 2020). One company, one cluster family, one vintage. Consequences in §9.

---

## 0. Conventions

| Convention | Rule |
|---|---|
| Sets | Calligraphic, e.g. $\mathcal{D}_i$ |
| Random variables | Uppercase, e.g. $R$, $X$ |
| Realisations | Lowercase with subscript, e.g. $R_i$, $x_i$ |
| Estimates | Hat, e.g. $\hat m(C)$ |
| Indices | $i,j$ requests, $r$ hazard step, $\beta$ bin |
| Units | Objects are counted; every byte-valued quantity is labelled |
| Ranges | Closed intervals $[a,b]$; half-open $[a,b)$ when a grid is partitioned |
| Dash policy | No em-dashes. Hyphens only inside compound terms |
| Index disambiguation | $k$ is always a key, $u$ is always a hazard step, $i,j$ are always requests |

Every symbol is defined at first use and collected in §10. Two rules are enforced by the docs linter: a symbol has exactly one meaning, and a symbol never carries two domains (objects versus bytes versus seconds).

---

## 1. Problem framing

### 1.1 Hit rate is a function, not a number

A cache of capacity $C$ objects serving a request stream has a **hit ratio** $\mathrm{HR}(C) \in [0,1]$: the fraction of requests served from the cache. Reporting "hit rate is 0.83" is a statement about one operating point of one deployment. It is not a property of the workload, because $\mathrm{HR}$ depends on $C$, and $C$ is chosen by the operator.

The object of study is the map

$$\mathrm{HR}: C \mapsto \mathrm{HR}(C), \qquad C \in \mathbb{N},$$

the **hit-ratio curve** (HRC), and its complement the **miss-ratio curve** (MRC)

$$m(C) = 1 - \mathrm{HR}(C).$$

Both names denote the same object in the literature. This document uses MRC for the curve and for its error metrics in §8, because errors are naturally expressed on the miss side, where the quantity is small and relative error is informative.

### 1.2 The decision the curve supports

Capacity is sized under two constraints: a latency service-level agreement (SLA) and a memory budget.

Let $L_{\text{hit}}$ and $L_{\text{miss}}$ be the mean latency of a hit and of a miss in seconds, with $L_{\text{miss}} > L_{\text{hit}}$. For a stream with hit ratio $\rho$, mean latency is

$$\mathbb{E}[L](\rho) = \rho L_{\text{hit}} + (1-\rho) L_{\text{miss}}.$$

An SLA $L^\star$ with $L_{\text{hit}} < L^\star < L_{\text{miss}}$ is met exactly when

$$\rho \ge \rho^\star := \frac{L^\star - L_{\text{miss}}}{L_{\text{hit}} - L_{\text{miss}}}.$$

Let $\bar s$ be mean object size in bytes and $B$ the memory budget in bytes. The sizing problem is

$$C^\star = \min \left\{ C \in \mathbb{N} \;:\; \mathrm{HR}(C) \ge \rho^\star \ \text{and}\ C\,\bar s \le B \right\}.$$

Three consequences follow, and each is a requirement on the statistics.

1. The feasible set may be empty. Proving infeasibility requires $\mathrm{HR}$ at the budget-limited capacity $C_B = \lfloor B/\bar s \rfloor$, not at the current capacity.
2. The optimum sits on a constraint boundary, so the decision depends on the local slope. A curve that is right on average but wrong near $C^\star$ yields the wrong $C^\star$.
3. The operator also asks about headroom: what happens at $1.5\times$ the current cache, and at $0.75\times$. These are counterfactual evaluations of the same function.

### 1.3 Why a point estimate is insufficient

A measured pair $(C_0, \mathrm{HR}(C_0))$ answers none of the three questions. It has no slope, so it cannot be extrapolated. Interpolating requires sampling at two capacities, which means deploying both. Fitting a parametric form to one point makes the answer a property of the assumed form, not of the workload.

A point estimate also hides its own uncertainty. Sizing decisions are made once and held for months, so the relevant quantity is a lower confidence bound on $\mathrm{HR}(C^\star)$, not a point value. Section 8 defines the interval; §5 explains why naive intervals are too narrow in the presence of censoring.

The requirement is therefore: estimate the whole function on a capacity grid $G$, with error bars, from a single observation of the stream at one capacity.

---

## 2. Reuse distance

### 2.1 Definition

The observation window contains $N$ requests indexed $i = 1,\dots,N$ in non-decreasing timestamp order, with key $k_i$ and timestamp $\tau_i$.

**Definition 1 (previous access).** For request $i$,

$$p(i) = \max\{ j < i : k_j = k_i \},$$

and $p(i) = \bot$ if no such $j$ exists.

**Definition 2 (reuse distance).** The reuse distance of request $i$ is

$$R_i = \left| \{ k_j : p(i) < j < i \} \right| \in \mathbb{N}_0, \qquad R_i = \infty \ \text{if} \ p(i) = \bot .$$

$R_i$ is the number of **distinct** keys requested in the open interval between the previous access to $k_i$ and the current one. It counts keys, not requests, and excludes both endpoints.

The support is $\mathbb{N}_0 \cup \{\infty\}$. The value $0$ means the same key was requested twice with nothing in between. The value $\infty$ is the **cold** case: the key has not been seen in the window.

Write $\mathcal{D}_i = \{k_j : p(i) < j < i\}$ for the set of distinct keys in the interval, so that $R_i = \lvert \mathcal{D}_i \rvert$. The set, not only its size, is needed for the byte-weighted variant in §3.3.

### 2.2 What it is not

Three quantities are routinely confused with $R_i$. They answer different questions, and only one indexes the LRU stack.

| Quantity | Definition | Unit | What it measures |
|---|---|---|---|
| Reuse distance $R_i$ | distinct keys in $(p(i), i)$ | objects | LRU stack depth of $k_i$ at request $i$ |
| Inter-arrival time $\Delta_i$ | $\tau_i - \tau_{p(i)}$ | seconds | temporal gap since last access |
| Reference distance $\tilde R_i$ | $i - p(i) - 1$ | requests | intervening requests, with multiplicity |

**Against inter-arrival time.** $\Delta_i$ is a property of the clock. Replay the stream at twice the rate and every $\Delta_i$ halves while every $R_i$ is unchanged. Occupancy is a function of what else arrived, not of how fast, so $\Delta_i$ is the wrong index for capacity. Under a Poisson stream of rate $\lambda$ over $K$ keys the two are related in expectation, $\mathbb{E}[R_i] \approx \lambda \Delta_i (1 - \text{collision terms})$, and that approximation is exactly what fails under bursts and diurnal load. $\Delta_i$ remains useful as a **feature** (§6), never as the identification variable.

**Against the reference distance.** $\tilde R_i$ counts every intervening request. A repeated key occupies one slot, not one slot per access, so $\tilde R_i \ge R_i$ always, with equality iff no key repeats inside the interval. Reference distance overestimates stack depth and, used as a surrogate, understates hit ratio. The bias is largest where caches are interesting: skewed key distributions with short-term bursts.

**Against "stack distance with repeats".** The phrase denotes $\tilde R_i$. Mattson's original stack distance is the distinct count, which is $R_i$. The two coincide only when no key is requested twice inside any reuse interval. This project uses the distinct count, and the example below shows the divergence on the third repeat.

### 2.3 Worked micro-example

Eight requests over three keys, $K=3$: `A B A C A B C B`.

| $i$ | $k_i$ | $p(i)$ | Distinct keys in $(p(i), i)$ | $R_i$ | $\tilde R_i$ | hit at $C=2$ | hit at $C=3$ |
|---:|---|---|---|---:|---:|:---:|:---:|
| 1 | A | $\bot$ | none | $\infty$ | $\infty$ | no | no |
| 2 | B | $\bot$ | none | $\infty$ | $\infty$ | no | no |
| 3 | A | 1 | $\{$B$\}$ | 1 | 1 | yes | yes |
| 4 | C | $\bot$ | none | $\infty$ | $\infty$ | no | no |
| 5 | A | 3 | $\{$C$\}$ | 1 | 1 | yes | yes |
| 6 | B | 2 | $\{$A, C$\}$ | 2 | 3 | no | yes |
| 7 | C | 4 | $\{$A, B$\}$ | 2 | 2 | no | yes |
| 8 | B | 6 | $\{$C$\}$ | 1 | 1 | yes | yes |

Hit counts: $C=2$ gives $3/8$, $C=3$ gives $5/8$. The three cold requests never hit, so $5/8$ is the ceiling at any capacity.

Request $i=6$ is the discriminating case. Its interval contains `A C A`: three requests, two distinct keys. Using $\tilde R_6 = 3$ would place $k_6$ at depth three and predict a miss at $C=3$, giving $4/8$ instead of $5/8$. The distinct count is what the stack actually holds.

Check against explicit LRU simulation at $C=3$: after `A B A` the cache holds $\{$A, B$\}$; `C` misses and enters, giving $\{$A, B, C$\}$; requests 5 to 8 all hit. Five hits. Definition and mechanism agree.

---

## 3. Identification

### 3.1 The identification theorem

**Theorem 1 (LRU identification).** Consider a cache of $C$ unit-size objects managed by LRU, with no expiration and no admission filtering. Request $i$ hits if and only if $R_i < C$.

*Proof.* Order the $C$ resident objects by recency of last access, most recent first, and assign each resident key its depth $0,\dots,C-1$. The keys accessed after $p(i)$ and before $i$ are precisely the keys more recently used than $k_i$. They are distinct by construction of $\mathcal{D}_i$, so they occupy exactly $R_i$ depth positions above $k_i$. Also $k_i$ was resident immediately after $p(i)$ and has not been accessed since, so only the arrival of other keys can have evicted it. It is evicted iff at least $C$ distinct keys arrived after it, that is iff $R_i \ge C$. $\square$

The theorem converts a dynamic process with $C$ interacting slots into a static property of each request, independent of $C$. Capacity appears only in the comparison.

**Corollary 1.** For a request with feature vector $x$, hit probability at capacity $C$ is a distribution function evaluated at $C-1$:

$$P(R \le C-1 \mid X = x) = F_R(C-1 \mid x).$$

The curve is one CDF and capacity is a query point into it. This is the formal statement of §1.1.

### 3.2 Assumptions and their violations

Theorem 1 is exact under five assumptions. Each is violated in production, and the violations differ in severity.

| Assumption | Status here | Consequence of violation | Treatment |
|---|---|---|---|
| LRU replacement | Approximately true for the cluster family studied; policy metadata is incomplete | A non-LRU policy retains objects LRU would evict, so the true curve lies above the LRU curve at small $C$ | Report as a policy-conditional curve, §9 |
| Unit object size | False; sizes span orders of magnitude | Small objects are over-penalised by object counting | Byte-weighted variant, §3.3 |
| No TTL | False for a subset of keys | Expiry removes objects that reuse distance believes resident, so the true hit ratio is lower | TTL-flagged keys excluded or reported separately |
| No admission filter | False if a TinyLFU-style filter is present | Rejected objects never enter, so a second request can miss with small $R$ | Treated as policy violation, not corrected |
| Single cache instance | False; the cluster is sharded | Each shard holds $C/n_{\text{shard}}$, and skewed keys concentrate on one shard | Capacity read as per-shard, or keys hashed across shards |

The first row is load-bearing. Everything downstream is conditional on it, and §9 forbids reading the result as a statement about the deployed policy.

### 3.3 Byte-weighted extension

Production caches are byte-bounded, and eviction is by bytes. Replace the count with a sum.

**Definition 3 (byte reuse distance).** With $s_k$ the size of key $k$ in bytes and $\mathcal{D}_i = \{k_j : p(i) < j < i\}$,

$$V_i = \sum_{k \in \mathcal{D}_i} s_k \ \text{[bytes]}, \qquad V_i = \infty \ \text{if} \ p(i) = \bot .$$

**Theorem 2 (byte-weighted identification).** Under byte-bounded LRU with variable sizes, request $i$ hits iff

$$V_i + s_{k_i} \le B.$$

*Proof.* The objects more recently used than $k_i$ are exactly $\mathcal{D}_i$, all distinct, all resident when the condition holds. LRU evicts in order of last use, so $k_i$ survives iff everything above it plus itself fits in $B$. If the sum exceeds $B$, the least recently used element of $\mathcal{D}_i \cup \{k_i\}$, which is $k_i$ itself, has been evicted. $\square$

Theorem 2 is not Theorem 1 with a rescaled capacity: the object-to-byte mapping depends on which keys lie in the interval, so the two curves differ in shape, not only in axis units. The project reports both. The object-count MRC is the clean theoretical object and connects to the literature; the byte-weighted MRC is what a capacity planner can act on. When they disagree materially, the report states the disagreement instead of choosing a winner.

A third variant, hit ratio weighted by bytes served rather than by request count, is out of scope and named in §9.

---

## 4. Discrete-time hazard

### 4.1 Definitions

$R$ is a discrete random variable on $\mathbb{N}_0 \cup \{\infty\}$ with covariate vector $X \in \mathbb{R}^P$.

**Definition 4 (discrete hazard).** For step $r \in \mathbb{N}_0$,

$$h_r(x) = P(R = r \mid R \ge r,\ X = x).$$

This is the conditional probability that reuse happens exactly now, given that it has not happened yet, for a request with features $x$.

**Definition 5 (survival).** $S_R(r \mid x) = P(R > r \mid X = x)$, with $S_R(-1 \mid x) = 1$.

**Definition 6 (distribution function).** $F_R(r \mid x) = 1 - S_R(r \mid x) = P(R \le r \mid X = x)$.

By Corollary 1, $\mathrm{HR}(C \mid x) = F_R(C-1 \mid x)$ and

$$\hat m(C) = \frac{1}{N} \sum_{i=1}^{N} \left[ 1 - \hat F_R(C-1 \mid x_i) \right].$$

The average is over requests, which is the correct denominator: the miss ratio is request-weighted, not key-weighted.

### 4.2 Product form

**Proposition 1.** $S_R(r \mid x) = \prod_{u=0}^{r} \left(1 - h_u(x)\right)$.

*Proof.* By the chain rule of conditional probability,

$$P(R > r \mid x) = \prod_{u=0}^{r} P(R > u \mid R > u-1,\ x),$$

with $P(R > -1 \mid x) = 1$. For integer-valued $R$ the event $\{R > u-1\}$ is $\{R \ge u\}$, so each factor is $P(R > u \mid R \ge u, x) = 1 - P(R = u \mid R \ge u, x) = 1 - h_u(x)$. $\square$

Consequently

$$F_R(C-1 \mid x) = 1 - \prod_{u=0}^{C-1}\left(1 - h_u(x)\right),$$

so the MRC at capacity $C$ is a product of $C$ conditional probabilities, one per stack slot. Two properties follow and are used later.

**Lemma 1 (monotonicity in capacity).** For fixed $x$, $C \mapsto F_R(C-1 \mid x)$ is non-decreasing, hence $C \mapsto \mathrm{HR}(C)$ is non-decreasing and $C \mapsto m(C)$ is non-increasing, for any hazard values in $[0,1]$.

*Proof.* Each factor $1 - h_u(x) \in [0,1]$, so the product is non-increasing in $r$, so $F_R$ is non-decreasing in $r$. Averages of non-decreasing functions are non-decreasing. $\square$

**Lemma 2 (concavity is not implied).** Nothing in the parametrisation forces $\mathrm{HR}$ to be concave in $C$. The empirical curve is often near-concave in $\log C$; the estimate need not be, and §8 measures the discrepancy rather than imposing it.

### 4.3 Why a hazard parametrisation rather than regression of the hit count

The alternative is to regress a hit indicator on $(x, C)$ directly. The hazard form is preferred for six reasons.

| Reason | Hazard form | Direct hit-count regression |
|---|---|---|
| Target | Binary and conditional, one row per step | Binary, conditioned on a design variable |
| Role of capacity | Absent from the model, enters only at evaluation | A feature, so the model must extrapolate in $C$ |
| Censoring | Enters through the risk set, no imputation | No natural place for a partial observation |
| Monotonicity in $C$ | Automatic, Lemma 1 | Must be imposed post hoc or learned |
| Statistical strength | Shared across $r$ by smoothness and monotonicity | Each capacity fitted from its own rows |
| Scoring rule | Logloss is proper for the factor it estimates | The fitted quantity is a marginal over a design choice |

The third row is decisive here. A request observed at the end of the window is known to have $R > \Gamma_i$, and that fact is informative. In the hazard form it enters the risk set and is then removed. In a capacity-indexed regression it is either dropped, which discards the information, or labelled a miss at every capacity, which is wrong for capacities below its eventual reuse distance.

The second row is what makes the curve a curve. Capacity is marginalised out of the model, so one fit produces $\hat F_R(\cdot \mid x)$ and every capacity is a lookup. The alternative needs a model per capacity, and models at neighbouring capacities are then free to cross.

### 4.4 Consistency check on the micro-example

The product form is verifiable by hand on the eight-request trace of §2.3. Treat the three cold requests as censored at step $3$, the largest observed finite distance.

| Step $r$ | At risk $n_r$ | Events $d_r$ | Hazard $h_r$ | Survival $S_R(r)$ | $F_R(r)$ |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 0 | 0.000 | 1.000 | 0.000 |
| 1 | 8 | 3 | 0.375 | 0.625 | 0.375 |
| 2 | 5 | 2 | 0.400 | 0.375 | 0.625 |
| 3 | 3 | 0 | 0.000 | 0.375 | 0.625 |

Reconstructed hit ratios are $F_R(C-1) = 0,\ 0.375,\ 0.625,\ 0.625$ for $C = 1,2,3,4$, against empirical values $0,\ 3/8,\ 5/8,\ 5/8$. Agreement is exact. Two features of the table are worth stating, because they recur at scale: the hazard is not monotone in $r$ (it rises from $0.375$ to $0.400$) and the curve is still monotone in $C$, which is Lemma 1 in miniature; and the risk set shrinks by exactly the event count at each step, $n_{r+1} = n_r - d_r$, which is the bookkeeping that §6.1 automates.

---

## 5. Right-censoring

### 5.1 The observation scheme

The trace ends at wall-clock time $T$. A request issued near $T$ may be reused after the window closes, and that reuse is never observed. This is **right-censoring**, and it is administrative: the cut is a property of data collection, not of the request.

**Definition 7 (event indicator).**

$$\delta_i = \begin{cases} 1 & \text{if } k_i \text{ is requested again at some } j > i \text{ with } \tau_j \le T,\\ 0 & \text{otherwise.} \end{cases}$$

**Definition 8 (censoring bound).** For a censored request, $\Gamma_i$ is the number of distinct keys requested in $(\max(i, p(i)), N]$, the distinct count accumulated from the request to the end of the window.

**Definition 9 (observed quantity).**

$$\tilde R_i = \begin{cases} R_i & \delta_i = 1,\\ \min(\Gamma_i,\ b_{M-1} - 1) & \delta_i = 0, \end{cases}$$

where $b_{M-1}$ is the last finite bin edge defined in §6.2, so a censored request carries the partial information $R_i > \tilde R_i$ rather than a fabricated value.

### 5.2 The independent-censoring assumption and its failure here

The standard assumption is that the censoring time is independent of the failure time given covariates:

$$\Gamma \perp R \mid X.$$

It fails here, and the failure is structural rather than incidental. The window ends at a fixed wall-clock time, so a request issued late has a short $\Gamma$ regardless of its key. If reuse distances and arrival times were independent that would be harmless. Under diurnal load they are not: requests issued at peak have shorter reuse distances than requests issued at 4 a.m., because interleaving traffic is heavier. Peak and off-peak requests are also spread unevenly across the window. Censoring probability therefore correlates with $R$ within strata of $X$ whenever $X$ does not encode the time-of-day effect.

The direction of the uncorrected bias is stated in §5.4. The response is twofold: include time-of-day and recent-arrival-rate features so that conditional independence is closer to true, and validate on a held-out time block whose censoring pattern differs from training.

### 5.3 Kaplan-Meier and the discrete-hazard likelihood

**Kaplan-Meier.** With $n_r$ the number of requests at risk at step $r$ (those with $\tilde R_i \ge r$) and $d_r$ the number with an event at step $r$,

$$\hat S_{\mathrm{KM}}(r) = \prod_{j=0}^{r} \left(1 - \frac{d_j}{n_j}\right), \qquad \hat F_{\mathrm{KM}}(r) = 1 - \hat S_{\mathrm{KM}}(r).$$

Ties are resolved by placing events before censorings at the same step, the conservative choice: a censored request stays in the risk set for that step. KM is the non-parametric maximum likelihood estimator and serves as the marginal, covariate-free reference curve in §7.

**Likelihood.** With $\theta$ the hazard parameters,

$$\mathcal{L}(\theta) = \prod_{i=1}^{N} \left[ h_{\tilde R_i}(x_i)^{\delta_i} \left(1 - h_{\tilde R_i}(x_i)\right)^{1-\delta_i} \prod_{u < \tilde R_i} \left(1 - h_u(x_i)\right) \right].$$

The three factors have distinct roles: the first is the event at the observed step, the second is survival through that step for a censored request, the third is survival through all earlier steps for every request. For $\delta_i = 1$ the second factor is a zeroth power and drops out; for $\delta_i = 0$ the first does. Both cases are one expression.

**Equivalence to a sequence of Bernoulli trials.** Expanding and regrouping by $(i,k)$ pairs gives

$$\ell(\theta) = \sum_{i=1}^{N} \sum_{u=0}^{\tilde R_i} \left[ y_{iu} \log h_u(x_i) + (1 - y_{iu}) \log\left(1 - h_u(x_i)\right) \right], \qquad y_{iu} = \delta_i \cdot \mathbb{1}[u = \tilde R_i].$$

This is the log-likelihood of independent Bernoulli trials, one per (request, step) pair, with a single positive label per request at its event step. The construction in §6.1 is exactly this expansion, which is why a binary classifier with logloss maximises the discrete-hazard likelihood without modification.

### 5.4 How the risk set removes the bias

Take a request censored at $\tilde R_i = 5$: the only fact known is $R_i > 5$. In the expanded form it contributes rows at steps $0,\dots,5$, all with $y = 0$. It appears in the denominator $n_u$ for those steps, teaching the model that requests like it do not fail early, and it disappears from step $6$ onward, where nothing is known. No value is imputed and no request is deleted. The information used is exactly the information available.

Ignoring censoring breaks this in one of three ways.

| Naive treatment | Mechanism | Direction on $\mathrm{HR}(C)$ |
|---|---|---|
| Drop censored requests | The sample truncates toward short reuse, since long reuse is what gets censored | Overestimate |
| Treat censored at $\tilde R_i$ as an event | Long reuse is rewritten as moderate reuse, moving mass from the tail into the body | Overestimate |
| Treat censored as never reusing, $\tilde R_i = \infty$ | Non-reusers stay in the risk set forever and never fail, inflating $S_R$ | Underestimate |

The first two dominate in practice because they are the convenient implementations. Both concentrate error in the tail of $F_R$, which is the region large capacities query, so the bias grows with $C$ over the range where the censored mass would have landed. The third is a survival package configured with the indicator reversed.

The magnitude is not assumed. §8 measures the uncorrected estimator against the exact curve on a deliberately truncated trace where the truncation point is known, and reports the gap. That experiment is the evidence that the correction matters at the capacities the decision uses.

---

## 6. Model

### 6.1 Risk-set construction

Input is the trace sorted by $(\tau_i, \text{sequence number})$, a total order that makes timestamps unique. Output is a set of rows $(i, r, y, x_i)$.

```
INPUT   requests 1..N with key k_i, size s_{k_i}, features x_i
        bin edges 0 = b_0 < b_1 < ... < b_{M-1}   (bin M-1 is the overflow bin)
OUTPUT  rows (r, y, x) and per-request (R_tilde, delta)

last <- empty map                       # key -> index of previous access
rows <- []
for i in 1..N:
    k <- key[i]
    if k in last:
        p <- last[k]
        D <- distinct_keys(prev_pos, p, i)     # Fenwick tree over prev positions
        R <- |D|;  delta <- 1
    else:
        R <- inf;  delta <- 0
    if delta == 0:
        R_tilde <- min(distinct_to_end(i), b_{M-1} - 1)
    else:
        R_tilde <- min(R, b_{M-1} - 1)         # right-truncation at overflow bin
    for r in 0 .. R_tilde:
        y <- 1 if (delta == 1 and r == R_tilde) else 0
        rows.append(r, y, x_i)
    last[k] <- i
```

Three implementation facts matter.

- **Distinct count.** Computed in $O(N \log N)$ with a Fenwick tree over previous-occurrence positions: $R_i$ is the number of positions $j \in (p(i), i)$ whose previous occurrence is $\le p(i)$. Memory is $O(K)$, one entry per live key.
- **Truncation.** Reuse distances at or beyond $b_{M-1}$ are pooled into the overflow bin and contribute no rows above it. $\hat F_R$ is therefore not identified beyond $b_{M-1}$, and the capacity grid must satisfy $\max G < b_{M-1}$. This is a stated domain restriction, not a defect.
- **Row count.** $\sum_i \tilde R_i$ rows, of order $N \bar R$, which reaches $10^{9}$ on a full trace. Mitigations are the grouped variant below, subsampling of long risk sets, and processing by time block.

**Grouped variant.** One row per (request, bin) pair instead of per step. If the hazard is constant within a bin of width $w_\beta$, the event probability in that bin is $1 - (1 - h_\beta)^{w_\beta}$ and so is the survival decrement. Binary logloss on grouped rows is an approximation to the exact likelihood with bias of order $w_\beta h_\beta^2$ per bin. The per-step expansion is the primary estimator; the grouped variant is a scaling option, and material disagreement between the two is reported.

**Single-row variant.** One row per request, with $r$ as a feature and a label only at the event step, drops the $\prod_{u < \tilde R_i}$ term. It is cheaper and biased, because surviving to step $u$ is itself evidence. It is retained as an ablation to quantify what the risk-set expansion buys.

### 6.2 Hazard binning

Bins are log-spaced because reuse distances are heavy-tailed and the capacity grid is itself log-spaced. With growth factor $\gamma > 1$,

$$b_\beta = \left\lceil \gamma^{\beta} \right\rceil - 1, \qquad \beta = 0,1,\dots,M-1, \qquad w_\beta = b_{\beta+1} - b_\beta,$$

with $b_M = \infty$ so that bin $M-1$ is the overflow bin.

$M$ and $\gamma$ are hyperparameters tuned jointly with the model. $M$ controls the bias-variance trade in the hazard step function: too few bins force a constant hazard across a decade of reuse distance, too many leave tail bins with almost no events. The default search covers $M \in [8, 64]$ on a log scale, with $\gamma$ implied by $M$ and the truncation point. The tuned values are reported.

Binning has three effects, all measured rather than assumed: it caps rows per request at $M$, which bounds cost; it smooths the hazard, reducing variance in the tail where events are scarce; and it quantises capacity resolution, since capacities inside one bin share a hazard unless the grouped survival is evaluated continuously within the bin.

### 6.3 GBDT rationale

The classifier is LightGBM with binary logloss on the expanded rows. Gradient-boosted decision trees suit this feature space for five reasons.

| Property | Why it matters here |
|---|---|
| Non-linear interactions | Reuse depends jointly on popularity, size, and time of day; the interaction is not additive |
| Mixed feature types | Counts, durations, ratios, and categorical tags coexist without transformation |
| Invariance to scaling | Split thresholds are invariant to monotone rescaling, so heavy-tailed counts need no normalisation |
| Native categorical handling | Namespace and shard identifiers are high-cardinality, so target statistics are avoided |
| Native missing values | Cold-start features are genuinely absent, not zero, and GBDTs route them |

The model is deliberately not a neural sequence model. The unit of observation is a request with tabular metadata, the row count is large, and interpretability of feature effects matters for the report. A sequence model over raw logs would be a different project and would make the censoring structure harder to represent.

**Features.** Every feature must be computable at serving time from state the cache already holds, and computed causally from requests strictly before $i$. Anything derived from the future is leakage and is rejected by the feature linter.

| Feature group | Examples | Rationale |
|---|---|---|
| Key identity proxies | key length, namespace prefix hash, type tag | Proxy for object class and access pattern |
| Object size | $s_{k_i}$, $\log s_{k_i}$ | Size correlates with reuse through the workload's object model |
| Key history | request count so far, time since last access, EWMA of $\Delta$ | Popularity and recent rhythm |
| Stream state | requests in last second, last minute, distinct keys in window, hour of day | Diurnal and burst context, and the strata that make §5.2 defensible |
| Frequency sketch | count-min estimate of key frequency | Cheap popularity proxy available at serving time |

### 6.4 Monotonicity

Two different monotonicities are routinely conflated, and the distinction determines what to constrain.

**In capacity.** By Lemma 1, $\hat m(C)$ is non-increasing in $C$ for any hazard values in $[0,1]$. This is automatic and needs no constraint. Observed non-monotonicity in $\hat m$ is an evaluation bug, not a model property.

**In the hazard step.** GBDT estimates of $h_r$ are not guaranteed monotone in $r$. On cache traces the hazard typically decreases with $r$, since long reuse is rarer conditionally on having survived, but an ensemble fit to finite data can produce local reversals.

The consequence usually attributed to this, that non-monotone $h_r$ makes the MRC non-monotone, does not follow. By Lemma 1 a reversal in $h_u$ cannot reverse the curve, because every factor remains in $[0,1]$. What a reversal does cause is real but different: inflated variance of $\hat S_R$ at large $r$, where estimation error accumulates multiplicatively through the product, and a hazard surface that is hard to read.

Non-monotonicity of the MRC does arise, from three other sources, each guarded separately.

| Source | Mechanism | Guard |
|---|---|---|
| Interpolation between bin edges | Piecewise-constant $\hat F$ interpolated or spline-smoothed across bins | Evaluate $\hat F$ at bin edges only; report a bin-aligned grid |
| Per-capacity recalibration | Separate calibration or separate models per capacity | Calibrate once, on the hazard scale |
| Direct CDF regression | Independent prediction of $F$ at each query capacity | Use the product form, never independent per-capacity heads |

The recommendation is to expose `monotone_constraints` on the step feature with decreasing direction as a tunable option, and to report the setting. Turned on, the constraint costs a little training likelihood and buys tail stability; turned off, the report states that the tail carries wider intervals. Either way, Lemma 1 is what guarantees a usable curve.

### 6.5 Training protocol and hyperparameters

Splitting is by contiguous time block, never at random: adjacent requests share keys, so a random split leaks the same key into train and test and inflates every metric. The protocol is train on all blocks but the last two, validate on the penultimate block for early stopping, test on the final block, and report all three.

| Hyperparameter | Symbol | Search range | Selection criterion |
|---|---|---|---|
| Number of bins | $M$ | 8 to 64, log scale | Validation IBS |
| Bin growth | $\gamma$ | 1.05 to 1.5 | Validation IBS |
| Leaves | none | 31 to 255 | Early stopping |
| Minimum rows per leaf | none | 100 to 5000 | Early stopping |
| Learning rate | none | 0.02 to 0.08 | Early stopping |
| Feature fraction | none | 0.6 to 1.0 | Validation IBS |
| L2 penalty | none | 0 to 10 | Validation IBS |
| Monotone constraint on $r$ | none | off, decreasing | Validation IBS, reported either way |

Tuning is Optuna with a fixed budget and a fixed seed set. Calibration is checked on the validation block before the test block is touched, and the test block is evaluated once.

---

## 7. Ground truth

### 7.1 Two independent computations

For a finite trace the MRC is a finite average, not an estimate:

$$m(C) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[R_i \ge C\right], \qquad m_B(B) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[V_i + s_{k_i} > B\right].$$

Two computations of this quantity are available, and they share no code.

| Route | Procedure | What it depends on |
|---|---|---|
| (a) Exact simulation | Replay the trace through an LRU simulator with capacity $C$; count misses | The eviction mechanism only |
| (b) Reuse-distance distribution | Compute $R_i$ for every request; count $R_i \ge C$ | Theorem 1 only |

**Proposition 2.** Routes (a) and (b) produce identical miss counts at every capacity, exactly, not up to sampling error.

*Proof.* By Theorem 1 the indicator $\mathbb{1}[R_i \ge C]$ equals the miss indicator of the simulation for each $i$, and both sums run over the same request set. Equality is term by term. $\square$

The proposition is what makes the pair a test rather than a comparison. Route (b) is a closed-form property of the trace; route (a) is an independent dynamic process. Any disagreement is an implementation defect, since no statistical error is possible.

### 7.2 The exact-first gate

No modelling result is admitted until three gates pass on a $10^5$ request prefix, with tolerance $0$ on counts.

| Gate | Test | Tolerance |
|---|---|---|
| G1, object variant | Simulated miss count equals $\sum_i \mathbb{1}[R_i \ge C]$ for all $C \in G$ | Exact integer equality |
| G2, byte variant | Size-aware simulated miss count equals $\sum_i \mathbb{1}[V_i + s_{k_i} > B]$ for all $B \in G_B$ | Exact integer equality |
| G3, warm-up alignment | Both routes use the same request population | Identical index sets |

A gate failure is a bug, reported with the first disagreeing request index. In practice the failures seen are ties in timestamps, which is why §6.1 fixes a total order by sequence number, and an off-by-one in whether the current request counts toward its own reuse distance.

### 7.3 Warm-up

The first requests in a window have no history. A first access of a key is cold, $R_i = \infty$, and can never hit, so it is a guaranteed miss at every capacity. Two treatments are legitimate and one is not.

| Treatment | Effect | Verdict |
|---|---|---|
| Include cold requests with $R_i = \infty$ | Correct: they are misses at all $C$ | Used |
| Drop the first $W$ requests from both routes | Correct if both routes drop the same set | Used for sensitivity |
| Drop cold requests but keep the rest | Hit ratio overestimated by $n_{\text{cold}}/N$ | Forbidden |

The default is the first row, since it keeps the full population and makes the cold-start cost explicit. The second row is the sensitivity check, with $W$ chosen so the cache has filled at least once, and the report states both curves when they differ by more than the §8 threshold.

### 7.4 What the model is competing against

Route (b) is exact for the observed trace, so no model can beat it on that trace. The model's value is elsewhere, and the report must not blur the distinction.

| Claim | Supported by |
|---|---|
| Reproduce the MRC of this trace | Route (b), exactly, no model needed |
| Predict the MRC of a future period from features | The model, validated on a held-out time block |
| Attribute hit ratio to key and stream properties | The model, via feature effects |
| Compress the trace into a compact curve generator | The model, measured by artifact size |
| Beat exact simulation on accuracy | Not claimed, and impossible by Proposition 2 |

This is why every metric in §8 compares the model to route (b), and why the comparison is a measure of approximation quality, not of discovery.

---

## 8. Evaluation

### 8.1 Capacity grid and error definitions

Let $G = \{C_1,\dots,C_{|G|}\}$ be a log-spaced capacity grid with $\max G < b_{M-1}$. Errors are defined on $G$ and reported as curves, not only as scalars, because a scalar hides where the error sits.

| Metric | Formula | Symbol | Unit |
|---|---|---|---|
| Miss ratio at a capacity | $\hat m(C)$ versus $m(C)$ | none | probability |
| Relative MRC error | $e(C) = \lvert \hat m(C) - m(C)\rvert / m(C)$ | $e(C)$ | dimensionless |
| Mean absolute error | $\mathrm{MAE} = \frac{1}{\lvert G\rvert}\sum_{C \in G}\lvert \hat m(C) - m(C)\rvert$ | $\mathrm{MAE}$ | probability |
| Maximum absolute error | $\mathrm{MAX} = \max_{C \in G}\lvert \hat m(C) - m(C)\rvert$ | $\mathrm{MAX}$ | probability |
| Normalised area between curves | $\mathrm{ABC} = \frac{1}{\log(C_{\max}/C_{\min})}\int_{C_{\min}}^{C_{\max}}\lvert \hat m(C) - m(C)\rvert\, d\log C$ | $\mathrm{ABC}$ | probability |
| Hit-ratio error at the operating point | $\lvert \mathrm{HR}(C^\star) - \widehat{\mathrm{HR}}(C^\star)\rvert$ | none | probability |
| Brier score at a capacity | $\mathrm{BS}(C) = \frac{1}{N}\sum_i w_i(C)\left(\hat p_i(C) - y_i(C)\right)^2$ | $\mathrm{BS}(C)$ | probability squared |
| Integrated Brier score | $\mathrm{IBS} = \frac{1}{\lvert G\rvert}\sum_{C \in G}\mathrm{BS}(C)$ | $\mathrm{IBS}$ | probability squared |
| Time-dependent AUC | $\mathrm{AUC}(C)$ from IPCW-weighted ROC of $\hat p_i(C)$ against $y_i(C)$ | $\mathrm{AUC}(C)$ | dimensionless |

Here $\hat p_i(C) = \hat F_R(C-1 \mid x_i)$, $y_i(C) = \mathbb{1}[R_i < C]$, and $w_i(C)$ is the inverse-probability-of-censoring weight

$$w_i(C) = \frac{\delta_i\,\mathbb{1}[\tilde R_i < C]}{\hat S_\Gamma(\tilde R_i^-)} + \frac{\mathbb{1}[\tilde R_i \ge C]}{\hat S_\Gamma(C^-)},$$

with $\hat S_\Gamma$ the Kaplan-Meier estimate of the censoring distribution and $r^-$ the step immediately before $r$. Censored requests with $\tilde R_i < C$ contribute nothing and are reweighted through the remaining terms, which is the standard Graf form.

**Why the ABC is normalised and taken in $\log C$.** Capacity decisions are relative (1.5x, 0.75x), so uniform weighting in $C$ would over-weight the large-capacity tail where the operator never operates. Integrating in $\log C$ and dividing by the log range makes the ABC an average absolute error in probability units, comparable across traces of different scale.

**Why relative error is capped.** At large $C$, $m(C)$ approaches $0$ and $e(C)$ diverges for any fixed absolute error. The report uses $e(C)$ only where $m(C) \ge 0.01$ and reports absolute error elsewhere, stating the rule once.

### 8.2 What failure would mean

A metric is only useful if its failure has an interpretation. Each row below names the defect it detects.

| Metric | What it catches | Failure threshold |
|---|---|---|
| $e(C)$ at any $C \in G$ with $m(C) \ge 0.01$ | Local shape error at a capacity the operator might choose | $> 0.10$ |
| $\mathrm{MAE}$ | Average miss-ratio displacement across the range | $> 0.02$ |
| $\mathrm{MAX}$ | A single badly estimated region, typically a bin edge | $> 0.05$ |
| $\mathrm{ABC}$ | Displacement weighted the way decisions are made | $> 0.02$ |
| Hit-ratio error at $C^\star$ | The decision itself is wrong | $> 0.02$ |
| $\mathrm{BS}(C)$ | Miscalibration of the per-request hit probability | $> 0.10$, or worse than the base-rate Brier |
| $\mathrm{IBS}$ | Calibration averaged over capacities | $> 0.08$ |
| $\mathrm{AUC}(C)$ | Failure to rank requests by reuse propensity | $< 0.70$ |

The thresholds are pre-registered. If a threshold is missed, the report states the miss, the affected capacities, and the diagnosis, and does not adjust the threshold.

**Directional reading.** A model that overestimates hit ratio is dangerous for sizing: it recommends a cache that misses the SLA. A model that underestimates is merely wasteful. The report therefore reports the signed error $\hat m(C) - m(C)$ alongside the absolute metrics, and the failure criterion at $C^\star$ is asymmetric: an overestimate of hit ratio by more than $0.01$ fails even though the absolute threshold is $0.02$.

### 8.3 Ablations that the metrics must separate

The evaluation is only complete if it isolates the three design choices that the statistics claim to matter.

| Ablation | Expected effect | Metric that shows it |
|---|---|---|
| No risk-set expansion, single row | Underestimates hit ratio, worst at small $C$ | $e(C)$ at small $C$ |
| Censoring ignored, censored requests dropped | Overestimates hit ratio, worst at large $C$ | signed error at large $C$ |
| Reference distance $\tilde R$ used instead of $R$ | Overestimates stack depth, underestimates hit ratio | $\mathrm{MAE}$ |
| Random split instead of time block | Metrics improve spuriously on test | gap between validation and test |

Each ablation is reported as a curve against the exact ground truth, so the reader sees the mechanism rather than a single number.

---

## 9. What this skeleton does not claim

### 9.1 Policy dependence

Theorem 1 is a statement about LRU. The fitted curve is the MRC an LRU cache of that capacity would see on this stream. It is not the curve of the deployed policy, and for admission-controlled policies such as W-TinyLFU it can be wrong in both directions at once: higher hit ratio at small capacity, because the filter protects the working set, and lower at large capacity, because the filter rejects items LRU would have kept. Any extrapolation to a different policy needs a policy-specific identification result, which this document does not provide.

### 9.2 Stationarity across time

The estimator conditions on features but assumes the conditional distribution $P(R \mid X = x)$ is stable across the evaluation window. Under diurnal load this is false in the mean: reuse distances at 3 p.m. and at 3 a.m. come from different distributions. Two consequences follow.

- The curve is an average over the window, weighted by the arrival pattern. Two windows with the same workload but different time-of-day mixes give different curves, and neither is wrong.
- The features in §6.3 that encode time of day and recent rate are load-bearing, not decorative. Without them, the conditional independence in §5.2 is implausible and the censoring correction is nominally applied but substantively weak.

The claim is conditional stationarity given the feature set, verified by the time-blocked test in §6.5. A drift test across blocks is reported in the results, not assumed away.

### 9.3 Feature availability

The model is only valid where its features exist at serving time. Features computed from the historical trace but unavailable to a live cache, such as future access counts or exact global key frequency, would make the curve unattainable in deployment. The feature linter rejects anything that reads an index greater than $i$, but it cannot detect a feature that is causally valid in the trace and unavailable in production, for instance a global count that a sharded cache does not hold. That check is manual and is recorded per feature.

A second gap: the model is trained on requests to keys it has seen. Cold keys have features defined by proxies only, and the model's behaviour on a genuinely new key population is untested. The report states the fraction of evaluation requests whose key was unseen in training.

### 9.4 Trace vintage and population

The data are from one company in 2020. Object size distributions, key skew, and traffic mix have all moved since. The document makes no claim of transfer to another operator, another year, or another workload class. What transfers is the method: reuse distance, the identification theorem, the hazard estimator, and the validation gates. The numeric curve does not.

### 9.5 Further limits

| Limit | Consequence |
|---|---|
| Byte-weighted variant ignores per-object metadata overhead and allocator fragmentation | The byte curve is optimistic at small object sizes |
| TTL is not modelled, only flagged | Keys with short TTL have a true hit ratio below the estimate |
| Sharding is not modelled | Per-shard capacity is assumed uniform, which is false under skew |
| Request-count weighting, not byte weighting | A byte-served hit ratio would weight large objects more heavily |
| No confidence band for the exact curve | Route (b) is exact, so intervals apply to the model only |
| One trace family | No variance estimate across clusters |

### 9.6 The one claim that is made

Subject to §3.2 and §9.1 to §9.5, a single fit of a discrete-hazard GBDT on request features reproduces the LRU miss-ratio curve of a held-out time block within the §8 thresholds, and the error is attributable to identifiable causes through the ablations in §8.3. That is the claim the report defends.

---

## 10. Symbol table

Every symbol in this document, its meaning, its domain, and the sections where it appears.

| Symbol | Meaning | Domain | Used in |
|---|---|---|---|
| $N$ | number of requests in the observation window | integer | 2, 5, 7, 8 |
| $i$ | request index | integer, $1..N$ | 2 to 8 |
| $j$ | summation or scan index over requests | integer | 2, 3, 5 |
| $k_i$ | key requested by request $i$ | string identifier | 2, 3, 6 |
| $K$ | number of distinct keys in the window | integer | 2, 6 |
| $\tau_i$ | timestamp of request $i$ | seconds | 2, 6 |
| $T$ | wall-clock end of the observation window | seconds | 5 |
| $p(i)$ | index of the previous access to $k_i$ | integer or $\bot$ | 2, 3, 6 |
| $\bot$ | sentinel for "no previous access" | symbol | 2 |
| $R_i$ | reuse distance, distinct keys since the previous access | objects, $\mathbb{N}_0 \cup \{\infty\}$ | 2 to 8 |
| $\tilde R_i$ | reference distance, intervening requests with multiplicity | requests | 2, 8 |
| $\Delta_i$ | inter-arrival time since the previous access to $k_i$ | seconds | 2, 6 |
| $\mathcal{D}_i$ | set of distinct keys in $(p(i), i)$ | set of keys | 3 |
| $V_i$ | byte reuse distance, summed sizes over $\mathcal{D}_i$ | bytes | 3, 7 |
| $s_k$ | size of object $k$ | bytes | 3, 6, 7 |
| $\bar s$ | mean object size | bytes | 1 |
| $C$ | cache capacity | objects | 1 to 8 |
| $C_B$ | budget-limited capacity, $\lfloor B/\bar s\rfloor$ | objects | 1 |
| $C^\star$ | operating capacity chosen by the sizing rule | objects | 1, 8 |
| $B$ | memory budget, or cache capacity in the byte variant | bytes | 1, 3, 7 |
| $G$ | capacity grid used for evaluation | set of integers | 1, 6, 7, 8 |
| $\lvert G\rvert$ | number of capacities in the grid | integer | 8 |
| $x_i$ | feature vector of request $i$ | $\mathbb{R}^P$ | 3 to 8 |
| $X$ | feature random variable | $\mathbb{R}^P$ | 3, 4 |
| $P$ | number of features | integer | 4, 6 |
| $r$ | hazard step index, a candidate reuse distance | integer, $\mathbb{N}_0$ | 4 to 8 |
| $u$ | dummy index over hazard steps in products and sums | integer, $\mathbb{N}_0$ | 4, 5 |
| $h_r(x)$ | discrete hazard at step $r$ for features $x$ | probability | 4 to 8 |
| $S_R(r \mid x)$ | survival function, $P(R > r \mid X = x)$ | probability | 4 |
| $F_R(r \mid x)$ | distribution function, $P(R \le r \mid X = x)$ | probability | 3 to 8 |
| $\hat F_R$ | estimated distribution function | probability | 4, 6, 8 |
| $m(C)$ | miss ratio at capacity $C$ | probability | 1, 3, 7, 8 |
| $\hat m(C)$ | estimated miss ratio at capacity $C$ | probability | 4, 8 |
| $\mathrm{HR}(C)$ | hit ratio at capacity $C$ | probability | 1, 3, 4, 8 |
| $\rho$ | hit ratio, as a free variable in the latency model | probability | 1 |
| $\rho^\star$ | hit ratio required to meet the latency SLA | probability | 1 |
| $L_{\text{hit}}$ | mean latency of a hit | seconds | 1 |
| $L_{\text{miss}}$ | mean latency of a miss | seconds | 1 |
| $L^\star$ | latency SLA | seconds | 1 |
| $\delta_i$ | event indicator, 1 if reuse is observed | $\{0,1\}$ | 5 to 8 |
| $\Gamma_i$ | censoring bound, distinct keys to the window end | objects | 5, 6, 8 |
| $n_r$ | number of requests at risk at step $r$ | integer | 5 |
| $d_r$ | number of events at step $r$ | integer | 5 |
| $\hat S_{\mathrm{KM}}(r)$ | Kaplan-Meier survival estimate at step $r$ | probability | 5 |
| $\hat F_{\mathrm{KM}}(r)$ | Kaplan-Meier distribution estimate | probability | 5 |
| $\hat S_\Gamma(r)$ | Kaplan-Meier estimate of the censoring distribution | probability | 8 |
| $\mathcal{L}(\theta)$ | likelihood of the hazard parameters | likelihood | 5 |
| $\ell(\theta)$ | log-likelihood | real | 5, 6 |
| $\theta$ | vector of hazard model parameters | real vector | 5 |
| $y_{iu}$ | label of the expanded row for request $i$ at step $u$ | $\{0,1\}$ | 5, 6 |
| $b_\beta$ | bin edge | objects | 5, 6 |
| $M$ | number of hazard bins, including the overflow bin | integer | 5, 6 |
| $\beta$ | bin index | integer, $0..M-1$ | 6 |
| $w_\beta$ | width of bin $\beta$ | objects | 6 |
| $h_\beta$ | within-bin hazard for bin $\beta$ | probability | 6 |
| $\gamma$ | growth factor of the log-spaced bin edges | real, $>1$ | 6 |
| $W$ | warm-up length excluded in the sensitivity check | requests | 7 |
| $n_{\text{cold}}$ | number of cold requests, those with $R_i = \infty$ | integer | 7 |
| $n_{\text{shard}}$ | number of cache shards | integer | 3 |
| $\lambda$ | request arrival rate | requests per second | 2 |
| $e(C)$ | relative MRC error at capacity $C$ | dimensionless | 8 |
| $\mathrm{MAE}$ | mean absolute MRC error over the grid | probability | 8 |
| $\mathrm{MAX}$ | maximum absolute MRC error over the grid | probability | 8 |
| $\mathrm{ABC}$ | normalised area between the estimated and exact MRC | probability | 8 |
| $\mathrm{BS}(C)$ | Brier score for hit prediction at capacity $C$ | probability squared | 8 |
| $\mathrm{IBS}$ | integrated Brier score over the grid | probability squared | 8 |
| $\mathrm{AUC}(C)$ | time-dependent AUC at capacity $C$ | dimensionless | 8 |
| $\hat p_i(C)$ | predicted hit probability of request $i$ at capacity $C$ | probability | 8 |
| $y_i(C)$ | realised hit indicator of request $i$ at capacity $C$ | $\{0,1\}$ | 8 |
| $w_i(C)$ | inverse-probability-of-censoring weight | dimensionless | 8 |
| $C_{\min}, C_{\max}$ | endpoints of the capacity grid | objects | 8 |

**Count: 71 symbols.** Two checks are run by the docs linter and must pass: every symbol above appears in the body, and no symbol appears in the body without an entry above.
