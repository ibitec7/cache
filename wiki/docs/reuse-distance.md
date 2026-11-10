# Reuse distance

## What it is

The reuse distance (stack distance) $d_i$ of request $i$ is the number of
*distinct keys* touched between the previous request for the same key and
request $i$. It is the exact precondition for an LRU hit: request $i$ hits in a
cache of $C$ objects if and only if $d_i < C$. First accesses have
$d_i = \infty$ by convention and always miss.

## Why it is needed here

One pass over the trace produces $d_i$ for every request, and from those
numbers the whole curve $m(C)$ follows by counting. That is the Mattson stack
algorithm ([Mattson et al. 1970](papers/mattson-1970.md)). It is exact, and it
is the reference the statistical model is scored against, so the definition has
to be pinned down before anything else: reuse distance in *distinct items*, not
in requests. The two differ, and the difference is a trap (see the micro-example).

## Formal statement

For request $i$ on key $k_i$, let $\pi(i)$ be the index of the previous request
for $k_i$, or $\bot$ if none. Then

$$
d_i = \begin{cases} \infty & \pi(i) = \bot \\ \left| \{ k_j : \pi(i) < j < i \} \right| & \text{otherwise} \end{cases}
$$

and the MRC under LRU is the empirical tail

$$
m(C) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{ d_i \ge C \}, \qquad H(C) = 1 - m(C).
$$

The stack algorithm computes all $d_i \ge 0$ in $O(N \log N)$ with an
order-statistics tree and $O(U)$ memory; the naive version is $O(NU)$.

| Symbol | Meaning | Domain |
|---|---|---|
| $i$ | request index | $1..N$ |
| $k_i$ | key of request $i$ | categorical |
| $\pi(i)$ | previous occurrence of $k_i$ | index or $\bot$ |
| $d_i$ | reuse distance, distinct keys strictly between | $\mathbb{N}\cup\{\infty\}$ |
| $C$ | capacity in objects | $[0,U]$ |
| $m(C)$ | miss ratio | $[0,1]$ |
| $H(C)$ | hit ratio | $[0,1]$ |
| $U$ | distinct keys in trace | $\mathbb{N}$ |
| $N$ | requests in trace | $\mathbb{N}$ |

Two properties used later:

1. **Monotonicity**: $m$ is non-increasing in $C$, so the curve is a valid CDF
   complement and can be stored as a sorted vector of $d_i$.
2. **Infinity mass**: the fraction with $d_i = \infty$ equals the compulsory
   ratio $\kappa$, the curve's floor.

## Worked micro-example

Trace A B A C B D. Walk it request by request, tracking the set of distinct keys
seen since each key's last access.

| $i$ | key | previous | keys strictly between | $d_i$ |
|---|---|---|---|---|
| 1 | A | none | - | $\infty$ |
| 2 | B | none | - | $\infty$ |
| 3 | A | 1 | B | 1 |
| 4 | C | none | - | $\infty$ |
| 5 | B | 2 | A, C | 2 |
| 6 | D | none | - | $\infty$ |

Note request 5: three requests elapsed (3, 4) between B at 2 and B at 5, but only
two *distinct* keys (A, C) intervene, so $d_5 = 2$, not 3. Using the elapsed
request count is the classic error; it is also why the hazard model must be
defined on distance, not on time ([Hazard model](hazard-model.md)).

Now count tails at four capacities:

| $C$ | $d_i \ge C$ for | misses | $m(C)$ | $H(C)$ |
|---|---|---|---|---|
| 1 | all six | 6 | 1.000 | 0.000 |
| 2 | $d \in \{\infty, 2\}$ | 5 | 0.833 | 0.167 |
| 3 | $d = \infty$ | 4 | 0.667 | 0.333 |
| 4 | $d = \infty$ | 4 | 0.667 | 0.333 |

Cross-check by direct simulation instead of counting: at $C=2$ the cache holds
two objects, so replay A,B,A,C,B,D and mark hits. A at 3 hits (B,A resident), C
at 4 evicts B, B at 5 misses, D at 6 misses: 1 hit, 5 misses, $m(2)=0.833$.
The two routes agree, which is the definition of correct reuse distance. At $C=3$
the cache holds three objects: A hits, C misses (A,B resident), B hits (A,C,B),
D misses: 2 hits, 4 misses, $m(3)=0.667$. Same as the table.

## Where this lives in the code

`src/mrc/reuse.py` (exact stack distance over a counter stack, plus the sampled
approximation), `src/mrc/mrc.py` (tail counting into the curve).
