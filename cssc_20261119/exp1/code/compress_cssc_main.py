"""Move derivations and implementation detail out of the CSSC main narrative."""

def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    a, b = text.index(start), text.index(end)
    if a >= b:
        raise ValueError(f"Invalid section order: {start}, {end}")
    return text[:a] + replacement.strip() + "\n\n" + text[b:]


THEORY = r"""
\section{Tariff decision framework}
\label{sec:tariff-decision-framework}

Let $M\geq0$ be the next monthly maximum with predictive law $F$, finite
mean $\mu_F$, and stop-loss transform
$H_F(t)=\mathbb{E}_F[(M-t)_+]=\int_t^\infty[1-F(x)]\,dx$.
The capacity, actual-demand, and contract-demand costs are, respectively,
\begin{equation}
\label{eq:tariff-costs-cssc}
C_F(\mathrm{cap})=r_cS,\quad
C_F(\mathrm{act})=r_d\mu_F,\quad
C_F(\mathrm{con},D)=r_d\{D+\kappa H_F(\alpha D)\},
\quad D\in[\delta S,S].
\end{equation}
Here $S$ is reference capacity, $r_c$ and $r_d$ are rates,
$\alpha$ is the overage threshold multiplier, and $\kappa$ is
the penalty multiplier. Thus tariff decisions depend on the mean and
the stop-loss curve, rather than on a tail quantile alone
(Table~\ref{tab:tariff-functionals}).

\subsection{Contract optimization and action regret}
\label{sec:contract-optimization}
The contract objective is convex. For an interior optimum when
$\kappa\alpha>1$, its threshold $\alpha D^*$ is a generalized quantile
at $p^*=1-(\kappa\alpha)^{-1}$, constrained to
$[\alpha\delta S,\alpha S]$. Equivalently, optimizing contract
demand amounts to minimizing a constrained Expected Shortfall
representation:
\begin{equation}
\label{eq:contract-constrained-es}
V_F(\mathrm{con})=
\frac{r_d}{\alpha}
\min_{t\in[\alpha\delta S,\alpha S]}
\left\{t+\frac{H_F(t)}{1-p^*}\right\}.
\end{equation}
The other optimized mode costs are $V_F(\mathrm{cap})=r_cS$ and
$V_F(\mathrm{act})=r_d\mu_F$. The oracle chooses the smallest of these
three values. For a fitted law $\widehat F$, define its selected
mode and contract level by the same optimization. Its oracle regret is
$R_F(\widehat a)=C_F(\widehat a)-V_F^*$, where
$V_F^*=\min_m V_F(m)$. If $\widehat m$ is contract demand, regret
contains both any cost of selecting the wrong mode and error in the
contract level:
\begin{equation}
\label{eq:within-contract-loss}
R_F(\widehat a)=
\{V_F(\widehat m)-V_F^*\}
+\{C_F(\widehat a)-V_F(\widehat m)\}.
\end{equation}
The second term is zero for the capacity and actual-demand modes.
These two components are computed for every simulated action.

\subsection{Decision stability}
For two laws $F,G$ with finite first moments, write
$W_1(F,G)=\int_0^\infty|F(x)-G(x)|\,dx$ for their one-dimensional
Wasserstein distance. The primitive-action envelope is
\begin{equation}
\label{eq:epsilon}
\varepsilon(F,G)=
\max\!\left\{r_d|\mu_F-\mu_G|,
r_d\kappa\sup_{D\in[\delta S,S]}
|H_F(\alpha D)-H_G(\alpha D)|\right\}.
\end{equation}
For an exact minimizer $\widehat a$ under $G=\widehat F$,
$R_F(\widehat a)\leq2\varepsilon(F,G)$; the guarantee does not
apply to an arbitrary action. If $F$ has a unique optimal mode with
gap $\gamma_F$, $2\varepsilon(F,G)<\gamma_F$ preserves that mode.
The envelope bounds mode values separately and can count common
forecast perturbations twice. A tariff-contrast bound retains their
cancellation.

\begin{proposition}[Tariff-contrast stability]
\label{prop:contrast-wasserstein-cssc}
Let $V_F(m)$ be the optimized mode values of
Equation~\eqref{eq:tariff-costs-cssc}, with a common feasible contract
set and $r_d,\kappa,\alpha>0$. For every distinct pair of modes
$m,n\in\{\mathrm{cap},\mathrm{act},\mathrm{con}\}$,
\[
|[V_F(m)-V_F(n)]-[V_G(m)-V_G(n)]|
\leq K_{m,n}W_1(F,G),
\]
where
\begin{equation}
\label{eq:contrast-constants-cssc}
K_{\mathrm{cap},\mathrm{act}}=r_d,\quad
K_{\mathrm{cap},\mathrm{con}}=r_d\kappa,\quad
K_{\mathrm{act},\mathrm{con}}
=r_d\max\{1,|1-\kappa|\}.
\end{equation}
If $m_F^*$ is unique, put
$\gamma_{F,m}=V_F(m)-V_F(m_F^*)>0$ for each competitor.
The mode selected under $G$ equals $m_F^*$ whenever
\begin{equation}
\label{eq:contrast-radius-cssc}
W_1(F,G)<\rho_F^{\mathrm{contrast}}
=\min_{m\ne m_F^*}\frac{\gamma_{F,m}}{K_{m,m_F^*}}.
\end{equation}
\end{proposition}
The capacity contrasts follow directly because capacity cost is
distribution-free. For the actual--contract contrast, define
$g_D(x)=x-D-\kappa(x-\alpha D)_+$.
Then $[V_F(\mathrm{act})-V_F(\mathrm{con})]/r_d
=\sup_D\mathbb E_F[g_D(M)]$. Its slopes are $1$ and
$1-\kappa$, so the Kantorovich--Rubinstein bound on each $g_D$,
followed by the supremum inequality, gives the third constant
\citep{RachevRomisch2002}. A positive oracle gap cannot change sign
under the stated radius; Appendix~S1 gives the full proof.
The radii have demand units (kW here). For the primary tariff
$\kappa=2$, the actual--contract constant is $r_d$, compared with
$r_d(1+\kappa)=3r_d$ from separate mode bounds. The resulting
numerical certificates are evaluated on the same decision rows in
Section~\ref{sec:results-stability}.
"""


EVALUATION = r"""
\section{Evaluation design and inference}
\label{sec:evaluation-design}
\label{sec:evaluation-inference}

\subsection{Controlled and stress experiments}
The primary experiment draws $10{,}000$ independent 24-month demand
histories from a Poisson--GPD event model. Each month has 200 bulk
events on $[45{,}000,80{,}000]$ kW, with scaled $\mathrm{Beta}(8,2)$
amplitudes, and $\mathrm{Poisson}(30)$ excess events above 80,000 kW
with GPD shape 0.15 and scale 12,000 kW. The monthly maximum is the
largest event. Specifically, $F_B$ is the CDF of
$45{,}000+35{,}000B$ for $B\sim\mathrm{Beta}(8,2)$, and $F_Z$
is the CDF of $80{,}000+Y$ for
$Y\sim\mathrm{GPD}(0.15,12{,}000)$. An analytic oracle gives
the monthly-maximum CDF,
$F_M(x)=F_B(x)^{200}\exp\{30(F_Z(x)-1)\}$, mean, stop-loss, and
optimal tariff action.

TAIL, EVENT-EMP, GEV, KDE, and EMP are fitted to the *same* history
and evaluated in nine fixed tariff cells: $(\alpha,u)=(1.05,0.620),
(1.05,0.630)$ and $\alpha=1.20$ with
$u\in\{0.400,0.450,0.455,0.625,0.630,0.635,0.700\}$,
where $u=\mu_F/S$. We use $r_d=36$, $r_c=22.5$ CNY/(kW month),
$\kappa=2$, and $\delta=0.40$. The cells span three oracle mode
regions and gaps of different sizes; they were fixed from oracle
geometry before method comparison. The rates are illustrative
coefficients for the computational design, not observed Korean tariffs.
The crossed-grid analysis reuses
1,000 histories over four $\alpha$ values and 19 utilization levels.
Additional designs raise the interior contract quantile target to
$0.90$, $0.95$, and $0.99$, change Poisson--GPD events to
negative-binomial counts with Weibull excesses, vary event clustering
and training length, and vary fitted support. Parameters and complete
results appear in Appendix~S1.

\subsection{Endpoints and inference}
\label{sec:controlled-mechanism}
Distributional endpoints are relative errors in the mean and
tariff-specific stop-loss, with $q_{0.99}$ error as a conventional
upper-tail comparator. Decision endpoints are mode correctness and
full-action regret, both absolute and divided by the oracle cost.
Regret is evaluated under the analytic oracle after each fitted law
has selected its action. Within each history, the nine cells are
equally weighted within their oracle-mode region; methods are paired
on history. Reported regional contrasts use paired Student $t$
95\% intervals over the 10,000 independent histories. Correlations
between distributional errors and regret use history-level averages;
ten correlation contrasts receive centered, studentized max-$|T|$
familywise intervals from 2,000 paired history bootstraps. Numerical
optimizer checks are in Appendix~S1.15.

\subsection{Gaussian monthly-maxima comparator}
\label{sec:gaussian-benchmark}
A nonnegative, zero-truncated Gaussian fitted by the sample mean and
standard deviation of monthly maxima supplies a simple parametric
comparator. We evaluate the 24- and 60-month prefixes of 100 paired
histories under each of the Poisson--GPD and negative-binomial--Weibull
mechanisms, with the 14 fixed EXP1 tariff cells. Regret is averaged
equally within tariff families and then across families; histories
are the independent units for paired $t$ intervals. This additional
comparison was specified after the original EXP1 results; its frozen
protocol, history hashes, code, and full contrasts accompany the paper.

\subsection{External manufacturing-load analysis}
\label{sec:external-manufacturing}
The external experiment uses public one-minute load records from
10 South Korean factories, March--September 2019
\citep{Lee2022KoreanFactories,LeeFigshareV9}. We reconstruct complete
15-minute averages, remove documented demand-response and 60-minute
rebound windows, and retain factory-months with at least 95\%
complete windows. Eight factories meet the fixed eligibility rule.
Four expanding-window origins forecast June--September monthly maxima.
The seven procedures are K-BLOCK (monthly block resampling), K-EMP
(empirical maxima), K-GEV, K-GTAIL (validated event tail), K-KDE,
K-POINT (a point forecast), and K-UGPD (unconditional GPD).
They are compared by CRPS, upper-tail weighted CRPS, and 0.95/0.99
pinball loss. The implemented upper-tail score is twice the trapezoidal
integral of pinball loss $\rho_\tau(y-q_\tau)$ over
$\tau\in\{0.900,0.905,\ldots,0.995\}$; its fixed grid and
normalization are specified in Appendix~S1.13. All model selection
and normalization use training
months only. For origin $o$, let $s_o$ be the mean of its training
monthly maxima; the normalized reference capacity is $S_o(u)=1/u$,
or $s_o/u$ in kW. Predictive distributions select actions in the
nine illustrative tariff cells; each is scored by the *hypothetical*
next-month charge divided by $s_o$. Methods share the same scale and
tariff within each factory-origin. The next-month observation is used
only for scoring, never for setting $S_o$.
Four origin differences are averaged within factory, then paired
$t$ intervals are constructed over eight factories. Processing,
validation, scoring, and fallback details are in Appendix~S1.13.
"""


def compress(text: str) -> str:
    text = replace_section(text, r"\section{Tariff and decision framework}",
                           r"\section{Distribution estimation and tariff actions}", THEORY)
    text = replace_section(text, r"\section{Evaluation design and statistical inference}",
                           r"\section{Results}", EVALUATION)
    text = text.replace(r"Table~\ref{tab:tariff-functionals}",
                        r"Equation~\eqref{eq:tariff-costs-cssc}")
    a = text.index(r"\captionof{table}{Tariff modes, within-mode decisions")
    a = text.rfind(r"\clearpage", 0, a)
    b = text.index(r"\clearpage", a + len(r"\clearpage"))
    text = text[:a] + text[b:]
    text = text.replace(r"\clearpage" + "\n" + r"\begingroup" + "\n" + r"\centering" + "\n" + r"\captionof{table}{Distributional",
                        r"\nolinenumbers" + "\n" + r"\clearpage" + "\n" + r"\begingroup" + "\n" + r"\centering" + "\n" + r"\captionof{table}{Distributional")
    text = text.replace(r"\resizebox{\linewidth}{!}{\begin{tabular*}{0.98\textwidth}",
                        r"\resizebox{0.94\linewidth}{!}{\begin{tabular}")
    text = text.replace(r"\end{tabular*}}}", r"\end{tabular}}}")
    text = text.replace(r"Table~\ref{tab:table_2}", r"Table~S16")
    a = text.index(r"\captionof{table}{Distributional and tariff-functional fidelity")
    a = text.rfind(r"\clearpage", 0, a)
    b = text.index(r"\clearpage", a + len(r"\clearpage"))
    text = text[:a] + text[b:]
    return text
