
$$
\begin{align*}
    X &\sim \distpois{\lambda} \\
    p_X(n) &= \frac{\lambda^n e^{-\lambda}}{n!} \\
    I_n &= \indicator{\text{$n$ appears}}
\end{align*}
\\[10pt]
\begin{align*}
    C &= \sum_{n = 0}^{\infty} I_n \\
    \expectation{C} &= \sum_{n = 0}^{\infty} \expectation{I_n} \\
    &= \sum_{n = 0}^{\infty} 0 \times (1 - p_X(n))^N + 1 \times (1 - (1 - p_X(n))^N) \\
    &= \sum_{n = 0}^{\infty} (1 - (1 - p_X(n))^N)
\end{align*}
\\[10pt]
\begin{align*}
    \variance{I_n} &= \expectation{{I_n}^2} - \expectation{I_n}^2 \\
    &= \expectation{I_n} (1 - \expectation{I_n}) \\
    &= (1 - p_X(n))^N (1 - (1 - p_X(n))^N)
\end{align*}
\\[10pt]
\begin{align*}
    \covariance{I_i, I_j} &= \expectation{I_i I_j} - \expectation{I_i} \expectation{I_j} \\
    &= \probability{I_i = 1 \land I_j = 1} - \expectation{I_i} \expectation{I_j} \\
    &= (1 - p_X(i) - p_X(j))^N - (1 - p_X(i))^N (1 - p_X(j))^N
\end{align*}
\\[10pt]
\begin{align*}
    \variance{C} &= \sum_{n = 0}^{\infty} \variance{I_n} + 2 \sum_{i < j} \covariance{I_i, I_j} \\
    &= \sum_{n = 0}^{\infty} \variance{I_n} + 2 \sum_{i = 0}^{\infty} \sum_{j = i + 1}^{\infty} \covariance{I_i, I_j} \\
    &= \sum_{n = 0}^{\infty} (1 - p_X(n))^N (1 - (1 - p_X(n))^N) + 2 \sum_{i = 0}^{\infty} \sum_{j = i + 1}^{\infty} \left[ (1 - p_X(i) - p_X(j))^N - (1 - p_X(i))^N (1 - p_X(j))^N \right]
\end{align*}
$$


