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
