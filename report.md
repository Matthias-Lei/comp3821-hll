\section{Testing}
\label{sec:shared-results}

The C++ tests use a shared harness. A test is a named function that asserts deterministic
properties or, for one estimator, checks a statistical property; the runner reports a
non-zero exit code if any check fails. Unit tests and benchmark experiments are separate:
the unit tests below use fixed seeds and are independent of the benchmark tables. The checks
described here are the ones implemented in the current source; a fresh run is required to
record pass/fail status.

Throughout this section, $n$ is the true number of distinct values, $B$ is the memory
budget in bits, and $T$ is the number of independent hash seeds (trials) used by a
statistical test. The meaning of the other memory symbols is sketch-specific: $m$ is the
bitmap size for Linear Counting and the Bloom filter, $r$ is the HLL register count, and
$k$ is the number of retained hashes for KMV or the number of hash functions for Bloom.

\subsection{Statistical Test: KMV}
Only KMV has a Monte Carlo bias test. It processes each of the Normal, Chi-Squared and Beta
observation streams at $B=8192$ bits ($k=128$) with $T=1000$ independently seeded runs. The
$i$-th seed is derived with splitmix64 from the trial index, so the runs are approximately
independent under the idealised uniform-hash model. There is no batching: the one-sample
$t$-statistic is formed directly from the $T$ per-run estimates, and the two-sided
$p$-value is computed as $p=\mathrm{erfc}(|t|/\sqrt{2})$ rather than through
$2(1-\Phi(|t|))$, which loses precision in the tail.

The three KMV tests use $\alpha=10^{-3}$. No multiple-testing correction is applied; with
three tests the family-wise error rate is about $0.3\%$. A large $p$-value is a pass/fail
gate, not a proof of unbiasedness. Linear Counting has a separate variance sanity check:
$m=4096$, $n=4096$, $T=400$, and the sample variance must lie within 35\% of $m(e-2)$;
it does not use a $p$-value. HLL and Bloom have deterministic unit tests only.

\subsection{Deterministic and Structural Tests}
The following properties are checked directly, not through a Monte Carlo bias test; the
HLL three-sigma check is a deterministic threshold on a fixed stream.
\begin{itemize}
    \item \textbf{KMV exactness for $n<k$:} the empty stream, $n=20$ with $k=21$,
    $n=11$ with $k=20$, and $n=2$ with $k=3$ must return the exact count.
    \item \textbf{Formula checks:} Linear Counting and the Bloom filter are checked against
    their closed-form estimates on a bitmap with a known number of set bits.
    \item \textbf{Small-stream rounding:} Linear Counting, Bloom and HLL are checked on
    small streams; the estimate must round to the true count for the smallest cases, and
    HLL additionally checks $n=50,100$ within three standard deviations of the
    Linear-Counting prediction.
    \item \textbf{Saturation:} Linear Counting and Bloom return \texttt{std::nullopt} when
    every bit is set, and Linear Counting returns a finite estimate when exactly one bit is
    empty.
    \item \textbf{Merge identity:} KMV and Linear Counting compare complete state with a
    direct build over the concatenated stream, including disjoint, overlapping and
    self-merge cases. HLL compares estimates only. Bloom has no merge operation.
    \item \textbf{Repeats and order:} Linear Counting compares bitmaps; Bloom and HLL
    compare estimates. KMV checks repeats but has no separate reversed-order test.
    \item \textbf{Validation:} Linear Counting rejects $m=0$ and mismatched $m$; KMV rejects
    $m=0$, $m<128$ and mismatched $k$; HLL rejects a mismatched register count. Bloom's
    $m=0$ path and HLL's too-small-budget path are implemented but not covered by the
    current unit tests.
    \item \textbf{Memory accounting:} each sketch's \texttt{memory\_bits()} is checked
    against its logical size; KMV reports $64k$ even though the heap and deduplication set
    use more physical memory.
\end{itemize}

\subsection{Coverage}
Table~\ref{tab:test-coverage} lists the checks that exist in the current suite. HLL++ is
excluded because its implementation and tests are not present in this checkout.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Property & LC & KMV & HLL & Bloom \\
        \midrule
        Formula / fixed-bitmap identity & \textbullet & -- & -- & \textbullet \\
        Exactness for $n<k$ & -- & \textbullet & -- & -- \\
        Small-$n$ rounded estimate & \textbullet & -- & \textbullet & \textbullet \\
        Monte Carlo bias test & -- & \textbullet & -- & -- \\
        Variance sanity check & \textbullet & -- & -- & -- \\
        Saturation $\rightarrow$ \texttt{nullopt} & \textbullet & -- & -- & \textbullet \\
        Merge identity & \textbullet\ state & \textbullet\ state & \textbullet\ estimate & no merge \\
        Self-merge & \textbullet & \textbullet & not tested & no merge \\
        Repeats & \textbullet & \textbullet & \textbullet & \textbullet \\
        Order independence & \textbullet & not tested & \textbullet & \textbullet \\
        Mismatched size rejected & \textbullet\ $m$ & \textbullet\ $k$ & \textbullet\ registers & no merge \\
        Constructor validation & \textbullet\ $m=0$ & \textbullet\ $m=0,64,127$ & not tested & not tested \\
        \texttt{memory\_bits} & \textbullet & \textbullet & \textbullet & \textbullet \\
        State equality & \textbullet & \textbullet & -- & -- \\
        \bottomrule
    \end{tabular}
    \caption{Checks implemented in the current C++ test suite. ``State'' means the complete
    sketch state is compared; ``estimate'' means only the estimate is compared.}
    \label{tab:test-coverage}
\end{table}

\subsection{Benchmark Conventions}
The benchmarks are a separate experiment from the unit tests. For Linear Counting, Bloom,
HLL and the LC/HLL merge benchmarks, \path{cpp/bench/bench.cpp} uses four budgets
(8192, 32768, 65536 and 131072 bits) and 64 seeds (1--64). The KMV benchmark is registered
with the four budgets but no seed range; the benchmark function reads a second argument
(seed) that this registration does not set, so it must not be described as a 64-seed
experiment. Dummy uses one seed. The benchmark has no HLL++ case, no Bloom merge case, and
no range-prefix or reversed-prefix mode.

Intervals in the result tables use the analysis script's 95\% $t$-interval for the mean:
$\bar{x}\pm t_{0.975,T-1}\,s/\sqrt{T}$, with $s$ the sample standard deviation. Where a
table's inputs are not present in the current checkout, the table is marked as a previous
run rather than presented as current output.


\section{Linear Counting}

\subsection{Introduction and Intuition}
Linear Counting (LC) uses an $m$-bit bitmap. Each hashed element sets one bit, and the
estimate is inferred from the fraction $V/m$ of bits that remain empty:
\[
    \hat{n} = -m \ln\left(\frac{V}{m}\right).
\]
When every bit is set ($V=0$), the estimate is undefined and the implementation returns
\texttt{std::nullopt}. The estimator is close to $n$ when bitmap collisions are negligible,
but it is not exact; the unit tests therefore require small-stream estimates to round to the
true count rather than match it exactly. For larger streams the relative bias tends to zero
as $n$ grows with $n/m$ fixed.

\subsection{Implementation}
The bitmap is stored as a packed \texttt{std::vector<uint64\_t>} with $m$ logical bits;
the final word may contain unused bits. \texttt{update} sets bit $h \bmod m$. \texttt{estimate}
counts the set bits with \texttt{std::popcount}, forms $V = m - \text{set bits}$, and
returns \texttt{nullopt} if $V=0$; otherwise it evaluates $-m\ln(V/m)$.
\texttt{merge} is a bitwise OR and requires equal $m$; \texttt{operator==} compares both
$m$ and the packed words. \texttt{memory\_bits()} returns $m$.

\subsection{Unique Tests}
The LC tests are deterministic except for one variance sanity check:
\begin{itemize}
    \item the closed-form estimate is checked on a bitmap with bits $0,\ldots,99$ set at
    $m=8192$;
    \item empty, five-element, twenty-element and repeated-value streams must round to the
    true distinct count (this is rounding, not exactness);
    \item repeats leave the bitmap unchanged, and reversing the stream leaves it unchanged;
    \item disjoint, overlapping and self-merge cases must equal the direct build over the
    concatenated stream, and merging a different $m$ must throw;
    \item a full bitmap returns \texttt{nullopt}; with exactly one empty bit the estimate is
    finite and equals $-m\ln(1/m)$; $m=0$ throws;
    \item the sample variance at $m=4096$, $n=4096$ must lie within 35\% of the
    asymptotic prediction $m(e-2)$ over $T=400$ trials. This is a sanity check, not a
    bias test.
\end{itemize}
There is no Monte Carlo bias test for LC in the current suite, and no saturation-rate
table is produced by the tests.

\subsection{Experimental Results}
The benchmark includes LC and LC merge with 64 seeds, but no benchmark result JSON is
present in this checkout. A current table would need to be filled from a fresh run rather
than copied from an earlier configuration; no LC numbers are reported here.


\section{K-Minimum Values}

\subsection{Introduction and Intuition}
K-Minimum Values (KMV) maps each element to a 64-bit hash and retains the $k$ smallest
distinct hashes. The implementation normalises the $k$-th smallest hash as
$U_{(k)} = H_{(k)}/(2^{64}-1)$; under the idealised uniform-hash model this is the $k$-th
order statistic of $n$ uniform points. For continuous uniform points,
$\mathbb{E}[U_{(k)}] = k/(n+1)$ and $U_{(k)} \sim \mathrm{Beta}(k, n-k+1)$, so
$\mathbb{E}[(k-1)/U_{(k)}] = n$. The estimator is therefore unbiased for $n \ge k$; its
relative standard error is approximately $1/\sqrt{k-2}$ for $k>2$, about $8.9\%$ at
$k=128$. When $n<k$ every hash is retained and the estimate is exactly $n$.

\subsection{Guarantees and Implementation}
The implementation gives the following guarantees:
\begin{itemize}
    \item the estimator is unbiased under the uniform-hash model for $n \ge k$;
    \item the estimate is exact when $n<k$;
    \item two sketches are mergeable only when they use the same $k$.
\end{itemize}

The sketch stores a max-heap of the retained hashes and a set for deduplication. The heap
supports peeking and popping the current maximum, while the set rejects duplicate hashes.
The number of retained values is $k = \lfloor B/64 \rfloor$, and the constructor requires
$B \ge 128$ ($k \ge 2$). The logical memory size reported by \texttt{memory\_bits()} is
$64k$; the heap and set use more physical memory than this logical budget. \texttt{merge}
requires equal $k$, throws on a mismatch, and does not modify the source sketch.

\subsection{Unique Tests}
The KMV tests check the exactness regime, the Monte Carlo bias test, deterministic merge
identity, and constructor/merge boundaries. The Monte Carlo test uses the Normal,
Chi-Squared and Beta observation streams; each has $n=1000$ and is processed at $B=8192$
bits ($k=128$) with $T=1000$ seeds. There is no batching, and the pass criterion is
$p>10^{-3}$ with the two-sided p-value from Section~\ref{sec:shared-results}.

The merge test compares complete sketch state with a direct build over the concatenated
stream. It covers disjoint halves, overlapping slices, self-merge, and the requirement that
the source sketch is unchanged. It also checks that a merge with a different $k$ throws.
No Monte Carlo merge-accuracy test is run; the state identity is the stronger check. A
$10\%$ per-run error bound at $k=128$ would be only about $1.1$ standard deviations, so it
is not used as a criterion.

\subsection{Test Results}
Table~\ref{tab:kmv_tests} lists the implemented checks. The table records what is asserted,
not a fresh pass/fail run; the runner is deterministic for the fixed seeds used here.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lll}
        \toprule
        \textbf{Category} & \textbf{Configuration} & \textbf{Assertion} \\
        \midrule
        Exactness & empty stream, $B=128$, $k=2$ & estimate $=0$ \\
        Exactness & $n=20$, $B=1344$, $k=21$ & estimate $=20$ \\
        Exactness & $n=11$, $B=1280$, $k=20$ & estimate $=11$ \\
        Exactness & $n=2$, $B=192$, $k=3$ & estimate $=2$ \\
        \midrule
        Monte Carlo & Normal/Chi-Squared/Beta, $n=1000$, $B=8192$, $k=128$, $T=1000$ & $p>10^{-3}$ \\
        \midrule
        Merge identity & disjoint halves of $0,\ldots,999$, $B=8192$, $k=128$ & state equals concatenated build \\
        Merge identity & overlapping slices, self-merge, source unchanged & state equals direct build \\
        Merge validation & different $k$ & \texttt{std::invalid\_argument} \\
        \midrule
        Boundaries & $B=0,64,127$; $B=128$; $B=8192$ & throws / accepted / \texttt{memory\_bits} \\
        Boundaries & repeated values; hash value $0$ & state unchanged; finite estimate \\
        \bottomrule
    \end{tabular}
    \caption{Implemented KMV checks. The Monte Carlo p-value is a gate at
    $\alpha=10^{-3}$; the exactness and merge checks are deterministic assertions.}
    \label{tab:kmv_tests}
\end{table}


\section{HyperLogLog}

\subsection{Introduction and Intuition}
At the heart of HyperLogLog (HLL) is estimating the number of distinct values $n$ using register maxima (Flajolet et al., 2007). We model distinct items' 64-bit hashes as independent and uniform. From this uniform hash, the position $\rho$ of its first $1$-bit (counting from the most significant bit) satisfies:
\[ \probability{\rho \geq r} = 2^{-(r - 1)}, \qquad 1\leq r\leq65-p. \]
This means a rank of $r$ is seen approximately once per $2^r$ distinct values and the largest rank amongst $n$ distinct values is about $\log_2 n$. 

However, a single maximum is a very unreliable and noisy estimate since the difference of one in the maximum rank would double or halve the estimate even for large streams (since standard deviation is constant in $\log_2 n$, so the relative error of $2^{\max\rho}$ does not shrink). HLL reduces this noise by using the first $p$ hash bits to select one of $r = 2^p$ registers, where each register stores the largest rank it has seen and then HLL combines these ranks through a harmonic mean to produce an estimate.

\subsection{Unique Tests}
\code{cpp/tests/test\_hyperloglog.cpp} checks the register count for the four budgets,
\texttt{memory\_bits} at the 8192-bit budget, the small-count estimates, repeats and
reordering, and one deterministic merge. For small counts $n=1,2,5,10$ the estimate must
round to $n$; for $n=50,100$ it must lie within three standard deviations of the
Linear-Counting prediction. The merge test splits a 200,000-key stream, checks that the
merged estimate equals the whole-stream estimate, bounds the relative error by
$4\cdot1.04/\sqrt{1024}$, and checks that a mismatched register count throws. It does not
construct ranks directly and does not perform a Monte Carlo bias test.

\subsection{Implementation}
The HLL sketch is at \path{cpp/src/hyperloglog.cpp}. At 1 KiB, HLL uses 10 bits of each 64-bit hash to select a register and the other 54 to calculate a rank. The largest possible rank is 55, when all 54 bits are zero. The register stores the largest rank it has seen, not the hash or its index. Across our four budgets, a rank therefore fits in six
bits. The two highest bits of each register byte remain zero and occupy memory without storing information.

\subsection{Experimental Results}
\textbf{Provenance.} The numbers in this subsection are from a previous benchmark run. The
streams for \(n=100{,}000\) and \(n=10{,}000{,}000\) and the corresponding JSON outputs are
not present in this checkout, so they are retained as historical measurements and are not
presented as output of the current test suite. [TODO: reproduce with the current benchmark.]

At \(n=100{,}000\), HLL's RMSE was 3.245\% [3.100, 3.383] at 1 KiB and 1.683\% [1.612, 1.752] at 4 KiB. The theoretical large-range RSE from the original HLL paper is approximately \(1.04/\sqrt{r}\), where \(r\) is the register count (not the bit budget) (\hyperref[flajolet2007]{Flajolet et al., 2007}). For these register counts, it gives 3.25\% and 1.63\%, respectively. Both lie within the measured RMSE
intervals, although RSE and RMSE differ when bias is present. Near the small-range
switch, the 16 KiB sketch had relative bias 2.414\% and RMSE 2.491\% at \(n=2.5m\). The raw estimate, not \(n\), controls the switch. Section~\ref{sec:shared-results} gives the experiment and intervals.

The current unit tests cover the register counts, small counts, repeats/order and one
deterministic merge; they do not construct rank cases. The current benchmark has no
range-prefix or reversed-prefix mode, so the earlier 6,144-prefix comparison cannot be
reproduced from this checkout and is not repeated here. Table~\ref{tab:hll-large} shows the
bias and RMSE at ten million distinct values from that previous run; it is not a current
result.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lrrr}
        \toprule
        Budget & Relative bias & Relative RMSE & HLL RSE approximation \\
        \midrule
        1 KiB & 1.256 [0.498, 2.014] & 3.264 [2.762, 3.698] & 3.250 \\
        4 KiB & 0.242 [$-$0.103, 0.587] & 1.392 [1.101, 1.632] & 1.625 \\
        8 KiB & 0.119 [$-$0.133, 0.371] & 1.008 [0.813, 1.171] & 1.149 \\
        16 KiB & 0.035 [$-$0.156, 0.226] & 0.759 [0.638, 0.863] & 0.813 \\
        \bottomrule
    \end{tabular}
    \caption{HLL error at \(n=10{,}000{,}000\), in percent. Previous run; intervals use
    64 seeds and the analysis script's t-interval method. Not reproduced in this checkout.}
    \label{tab:hll-large}
\end{table}

\section{HyperLogLog++}

\textbf{Status.} The HLL++ implementation and
\path{cpp/tests/test_hyperloglog_plus_plus.cpp} are not present in this checkout. They
exist on an unmerged branch and still use the older test style, so HLL++ is not part of the
current coverage matrix in Section~\ref{sec:shared-results}. The material below is from
that branch and has not been re-run against the current harness. [TODO: merge the branch
and rerun its tests and benchmark.]

\subsection{Introduction and Intuition}
HLL++ extends HLL to improve small-cardinality accuracy and space use
(\hyperref[heule2013]{Heule et al., 2013}).

\textbf{Sparse representation.}

\textbf{Dense correction.}

\subsection{Unique Tests}
On that branch, \path{cpp/tests/test_hyperloglog_plus_plus.cpp} covers:
\begin{itemize}
    \item sparse prefix counts and repeated inputs at the conversion boundary
    \item preservation of ranks during sparse-to-dense conversion
    \item bias-table outputs and sparse and dense merges, including a case
    where direct and merged sketches use different representations.
\end{itemize}
These checks are not part of the current checkout's test suite.

\subsection{Implementation}


\subsection{Experimental Results}
The results below are from the unmerged HLL++ branch and are not reproduced in this
checkout; they are retained for context only. On that branch, the unit tests cover the
sparse and dense merge cases. The boundary test shows how, when merging, the direct and
merged sketches can use different representations near conversion. Once both sketches were
dense in that test, their estimates agreed.

The branch benchmark gave equal merged and direct estimates in 6,143 of 6,144 range-prefix
trials. The one difference occurred at 1 KiB and \(n=269\), near sparse-to-dense
conversion. Reversing the prefixes also changed one HLL++ estimate, at the same budget,
count and seed. Both checks use 64 seeds. They compare estimates, not complete internal
states. These are branch results; the current benchmark contains no HLL++ case.

Table~\ref{tab:hllpp-10000} gives HLL++ results at 10,000 distinct values from that branch.
All trials used dense mode. At 4 KiB, HLL++ reduced the measured bias from HLL's 1.439\% to
0.059\% and RMSE from 2.972\% to 1.344\%. At 1 and 16 KiB, the two implementations returned
identical estimates for all 1,024 seeds.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lrr}
        \toprule
        Budget & Relative bias & Relative RMSE \\
        \midrule
        1 KiB & 0.061 [$-$0.126, 0.249] & 3.060 [2.926, 3.189] \\
        4 KiB & 0.059 [$-$0.023, 0.141] & 1.344 [1.286, 1.401] \\
        8 KiB & 0.054 [$-$0.001, 0.108] & 0.894 [0.856, 0.930] \\
        16 KiB & $-$0.018 [$-$0.055, 0.019] & 0.604 [0.578, 0.630] \\
        \bottomrule
    \end{tabular}
    \caption{HLL++ error at \(n=10{,}000\), in percent, from the unmerged branch; 95\%
    intervals over 1,024 branch seeds. Not reproduced in this checkout.}
    \label{tab:hllpp-10000}
\end{table}

The sparse-to-dense transition affects accuracy. At 1 KiB on the range
stream, HLL++ RMSE rose from 0.012\% at \(n=256\) to 2.218\% at \(n=269\).
The fraction still sparse fell from 100\% to 4.8\%. These checkpoints do
not identify the exact transition for each seed. The sparse estimate follows
Linear Counting on \(2^{25}\) prefixes under the uniform-hash model. It is
not exact, even when no prefix collisions occur in a sampled run.
At ten million distinct values, on that branch HLL++ and HLL returned the same estimates
for all 256 seed and budget pairs. Table~\ref{tab:hll-large} therefore also describes the
branch's HLL++ at that count; this is not a current-checkout result. The original HLL RSE
approximation there is consistent with the measured HLL++ RMSE intervals. It is not a
separate accuracy guarantee for HLL++ at smaller counts or near conversion.


\section{Bloom filter}
\subsection{Introduction and Intuition}
Bloom filters were originally designed for membership testing. Using an $m$-bit array and $k$ hash functions, it trades away some accuracy at the cost of a small false positive rate and massive memory/space savings. 

However, the mathematical properties of the Bloom filter enable it to be repurposed for cardinality estimation via the bit-density method. To derive the estimator, consider the probability that a specific bit remains $0$. A single hash leaves the bit untouched with probability $1 - 1/m$. With $k$ independent hashes, the probability that it remains untouched by one element is \((1 - 1/m)^k\). After $n$ distinct elements, the probability that the bit is still \(0\) is \((1 - 1/m)^{kn}\), which approximates to \(e^{-kn/m}\) for large $m$. 

Let $V$ be the observed number of empty bits. Substituting the observed fraction \(V/m\) into the probability equation and taking the natural logarithm yields
\[ \ln\left(\frac{V}{m}\right) = -\frac{kn}{m}. \]
Rearranging for \(n\) gives the generalised estimator
\[ \hat{n} = -\frac{m}{k} \ln\left(\frac{V}{m}\right). \]
This is conceptually identical to Linear Counting (\(k=1\)). At the same empty fraction the
Bloom estimate is $1/k$ of the LC estimate; the saturation point itself depends on the
stream, so the tests only assert that a full bitmap returns \texttt{nullopt}. Because every
element sets $k$ bits simultaneously, the Bloom filter saturates faster than a traditional
bitmap, serving as an interesting baseline for the space-accuracy trade-off.

\subsection{Unique Tests}
The Bloom tests are deterministic. The closed-form estimator is checked at $m=8192$ with
$k=3$ and exactly 100 set bits; empty and small streams must round to the true distinct
count; repeats and reordering must leave the estimate unchanged; a full bitmap at $m=64$
and $m=100$ must return \texttt{std::nullopt}; and \texttt{memory\_bits()} must equal $m$.
The Bloom filter has no merge operation in this implementation, and there is no Monte
Carlo bias test. The constructor's $m=0$ path is implemented but not covered by a unit
test.

\subsection{Implementation}
To implement the Bloom filter for cardinality estimation, we utilized a bit-packed \texttt{std::vector<uint64\_t>} for cache-friendly memory access and hardware-level \texttt{std::popcount} for fast bit counting.

Computing $k$ fully independent hash functions is computationally expensive. Instead, we
apply the Kirsch-Mitzenmacher double-hashing construction to approximate $k=3$ independent
hash functions from a single 64-bit input. Splitting the incoming hash into two 32-bit
halves, $h_1$ and $h_2$, gives the indices
\(h_i = (h_1 + i \cdot h_2) \bmod m\). This reduces CPU cost but is only an approximation
to independent hashing. Because $h_2$ is used modulo $m$, the degenerate case
$h_2 \equiv 0 \pmod m$ collapses all $k$ indices to one; for uniform $h_2$ this happens
with probability about $1/m$ (exactly $2^{-13}$ at $m=8192$). The current tests do not
cover this degenerate case. If the bitmap fully saturates ($V=0$), the estimator returns
\texttt{std::nullopt}.

\subsection{Experimental Results}
The benchmark includes Bloom with 64 seeds, but no benchmark result JSON is present in this
checkout. No Bloom error table is reported here; a current table would need a fresh run
rather than a copy of an earlier configuration. [TODO: reproduce with the current benchmark.]

