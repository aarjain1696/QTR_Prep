# Volatility Surfaces, Explained from Zero
### A companion to `Week1-Volatility_Surface.ipynb`

**Who this is for.** You work in finance but have never built a vol surface, calibrated SABR, or
inverted Black–Scholes. By the end you should be able to (1) explain *what every cell does and why*,
(2) reproduce the intuition on a whiteboard for a quant, and (3) know exactly what you'd change to run
it on a real desk.

**How to read it.** Each section follows the same rhythm:

- **The intuition** — the idea in plain English, before any symbols.
- **The math** — the formula, then a line-by-line reading of what it *says*.
- **What the code does** — how the notebook implements it.
- **🏦 In the real world** — where this diverges from desk practice, cross-referenced to
  `Week1-Volatility_Surface_Review.md`.
- **🎤 Explain it to a quant in one breath** — the one or two sentences that prove you get it.

> **The single most important idea in this whole notebook.** "Volatility" is not one number. There
> are **three** different things people call volatility and you must never mix them up:
>
> | Name | Looks | Measure | Question it answers |
> |---|---|---|---|
> | **Realized** vol | backward | physical (ℙ) | What did the stock *actually* do? |
> | **Implied** vol | forward | risk-neutral (ℚ) | What do option *prices* imply about the future? |
> | **Model** vol (GARCH/SABR/Heston) | either | depends | Our attempt to *describe or forecast* the other two. |
>
> "ℙ vs ℚ" (physical vs risk-neutral measure) just means: ℙ is the real world where you estimate from
> history; ℚ is the pricing world where everything is discounted at the risk-free rate and prices are
> expectations. Option prices live in ℚ. Your return history lives in ℙ. The gap between them is not a
> mistake — it's where the money is (Section 6).

---

## The 30-second picture: what is a volatility surface?

Black–Scholes says every option on a stock should be priced off **one** volatility number. If that
were true, you could back out that number from any option's market price and always get the same
answer. You don't. The implied volatility you extract **depends on the strike and the maturity** of
the option you looked at.

Plot that dependence in 3D — implied vol on the vertical axis, **strike (or moneyness)** on one
horizontal axis, **time to maturity** on the other — and you get a **surface**. It is wrinkled, not
flat:

- The wrinkle *across strikes* is the **smile / skew**.
- The drift *across maturities* is the **term structure**.

The entire notebook is: build this surface from real AAPL option prices, then show the models (SABR,
Heston) that desks use to describe its shape and fill in the gaps between quoted strikes.

---

## Section 0 — Setup

**What the code does.** Imports the scientific Python stack and pins a few conventions:

- `TICKER = "AAPL"`, `TODAY = today's date`.
- `N_DAYS_YEAR = 252` — the number of **trading** days in a year, used to *annualize* volatility.
- `brentq` (a 1-D root finder), `least_squares` (for SABR calibration), `griddata` (to interpolate a
  surface from scattered points), `plotly` (interactive 3D).

Nothing conceptual here — but note the **252**. Volatility is measured per-day from trading-day
returns, so to express it "per year" you scale by trading days, not calendar days. Hold that thought;
it collides with a *different* convention later (see Section 6's real-world note).

---

## Section 1 — The risk-free discount curve

### The intuition
Money tomorrow is worth less than money today. An option pays off in the future, so its value today is
the **discounted** value of its expected payoff. "Discounting" needs an interest rate. But there isn't
*one* rate — a 3-month rate differs from a 30-year rate. That whole set of rates-by-maturity is the
**term structure** (or **yield curve**). To price options at different expiries, you need the rate
*appropriate to each expiry*.

### The math
Every option price is a **discounted risk-neutral expectation**:
$$ C = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\big[(S_T-K)^+\big]. $$
Read it as: *the price today = (a discount factor) × (the average payoff in the pricing world)*. The
$(S_T-K)^+$ is the call payoff (`max(stock − strike, 0)`). The $e^{-rT}$ drags that future average
back to today.

The notebook converts a quoted annual yield $y$ into a **continuously compounded** rate so that the
discount factor is a clean exponential:
$$ r_{cc} = \ln(1+y), \qquad P(0,T) = e^{-r_{cc}\,T}. $$
Continuous compounding is just the mathematically convenient limit of compounding "infinitely often";
it makes the algebra in Black–Scholes (which is full of exponentials) tidy.

### What the code does
- `load_treasury_curve()` pulls US Treasury constant-maturity yields from **FRED** (e.g. `DGS3MO`,
  `DGS10`) — these are the market's risk-free rates by tenor. If no FRED key is set, it falls back to
  yfinance Treasury indices.
- It converts each percent yield to a continuously compounded decimal with `np.log1p` (= `ln(1+y)`).
- `risk_free_rate(T)` **linearly interpolates** that curve to any maturity `T` you ask for.
- `discount_factor(T)` returns $e^{-r(T)T}$.

So given any option maturity, you can now get "the right interest rate" and "the right discount
factor." That's the foundation everything else stands on.

### 🏦 In the real world
*(Review §3, bullets 1–2)*
- **Par yields are not zero rates.** The notebook treats a quoted Treasury yield as if it were the
  pure discount (zero) rate. A desk **bootstraps** true zero rates from instruments — and uses the
  **OIS/SOFR** curve, not Treasuries, as the genuine risk-free discounting curve for derivatives.
- **Compounding convention.** Treasury CMT yields are **bond-equivalent (semi-annual)**, so the
  cleaner conversion is $r_{cc} = 2\ln(1+y/2)$, not $\ln(1+y)$ (~4 bp difference at 4% — small, but
  it's the kind of thing a quant will ask you about).
- **Interpolation space matters.** Linear interpolation *in the zero rate* can produce kinked or even
  negative *forward* rates. Desks interpolate in log-discount-factor or forward space for smoothness.

### 🎤 Explain it to a quant in one breath
"An option price is a discounted ℚ-expectation, so I need a discount factor per maturity. I build a
continuously-compounded zero curve and interpolate it; in production I'd bootstrap zeros off OIS/SOFR
rather than treating par yields as zeros."

---

## Section 2 — Realized (historical) volatility

### The intuition
The simplest meaning of "volatility": how much did the stock actually bounce around? You measure it
from the **dispersion of past returns**. Crucially, when you plot it through time it is *not constant*
— markets have calm stretches and stormy stretches. Vol **clusters**: a big move today makes a big
move tomorrow more likely. That single empirical fact is the seed of everything that follows (GARCH,
the smile, stochastic vol).

### The math
Use **log returns** $r_t = \ln(S_t / S_{t-1})$. Why logs and not simple percentage changes?
- They're **time-additive** (a week's log return is the sum of the daily ones).
- They're the natural language of **geometric Brownian motion**, the model under Black–Scholes, where
  log returns are normally distributed.

Realized volatility is the standard deviation of those returns, **annualized**:
$$ \sigma_{\text{realized}} = \sqrt{N}\;\operatorname{std}(r_t), \qquad N = 252. $$
Why the $\sqrt{N}$? For independent returns, **variance grows linearly with time** (variance of a sum
= sum of variances), so a daily variance $\times 252$ gives annual variance; take the square root to
get back to volatility. Volatility scales with the **square root of time** — the famous "√t rule."

### What the code does
- Downloads 3 years of AAPL closes (`auto_adjust=True`, so prices are adjusted for splits/dividends —
  correct for computing returns).
- Computes daily log returns.
- Computes **rolling** 21-day (~1 month) and 63-day (~3 month) realized vol, annualized, and plots
  them. You can *see* the clustering: the lines are far from flat.

### 🏦 In the real world
*(Review §3, "Close-to-close realized vol only")*
- This is the **close-to-close** estimator — it only uses the closing price and throws away the day's
  high/low. It's correct but **noisy**. Desks often use **range-based estimators** (Parkinson,
  Garman–Klass, Yang–Zhang) that exploit the OHLC bar you already downloaded and are several times more
  statistically efficient. Good lesson: *estimating* vol and *modeling* vol are different problems.

### 🎤 Explain it to a quant in one breath
"Realized vol is the annualized standard deviation of log returns, scaled by √252 because variance is
additive in time. The rolling estimate visibly clusters, which is why a single constant σ can't be
right — and motivates GARCH."

---

## Section 3 — GARCH(1,1): modeling and forecasting realized vol

### The intuition
If volatility clusters, then today's variance carries information about tomorrow's. GARCH is the
workhorse model that encodes exactly that: **tomorrow's variance is a blend of (a) a long-run average,
(b) yesterday's surprise, and (c) yesterday's variance.** It naturally produces vol that spikes after
a shock and then *decays back* toward a long-run level — i.e. **mean reversion**.

### The math
$$ \sigma_t^2 = \underbrace{\omega}_{\text{baseline}} + \underbrace{\alpha\,\epsilon_{t-1}^2}_{\text{reaction to last shock}} + \underbrace{\beta\,\sigma_{t-1}^2}_{\text{persistence}}. $$
- $\alpha$ — how strongly a fresh shock ($\epsilon_{t-1}$ = yesterday's return surprise) bumps vol up.
- $\beta$ — how much of yesterday's variance carries over (memory/persistence).
- $\alpha + \beta$ — **total persistence**. Close to 1 ⇒ shocks die out very slowly.
- **Long-run (unconditional) variance** = $\dfrac{\omega}{1 - \alpha - \beta}$. This is the level vol
  reverts to. (It only exists if $\alpha+\beta<1$.)

Because the model knows how vol decays, you can roll it forward to get a **forecast term structure of
volatility**: expected average vol over the next 1 day, 1 month, 1 year… mean-reverting from today's
level toward the long-run level.

### What the code does
- Fits `arch_model(..., vol="GARCH", p=1, q=1, dist="normal")` on returns **in percent** (the `arch`
  library's convention, purely for numerical stability).
- Prints the fitted $\omega, \alpha, \beta$, the persistence $\alpha+\beta$, and the implied long-run
  annualized vol.
- Plots the in-sample **conditional volatility** (the model's day-by-day vol) against realized vol.
- Forecasts variance out 252 days and converts it to a **term-structure curve**: term vol to horizon
  $h$ = $\sqrt{\frac{1}{h}\sum_{i=1}^h \text{(daily var}_i) \times 252}$. This `GARCH_TERM` curve is
  reused in Section 6 to compare against implied vol.

### 🏦 In the real world
*(Review §3, "GARCH(1,1) with Gaussian innovations and no leverage term"; §3, "P vs Q")*
- **Gaussian innovations underestimate tail risk.** Equity returns are fat-tailed; switch to
  **Student-t** (`dist="t"`).
- **Plain GARCH is symmetric — it can't see the leverage effect.** Empirically, equity vol jumps more
  on *down* days than up days. That asymmetry is the **time-series cousin of the very skew** you'll see
  in the option surface. **GJR-GARCH** or **EGARCH** capture it and are essentially one-line changes.
  This is a beautiful point to raise with a quant: realized-vol asymmetry (ℙ) and implied-vol skew (ℚ)
  are two windows onto the same economic fact.
- **GARCH lives in ℙ, implied vol lives in ℚ.** When you compare them in Section 6, remember the GARCH
  forecast is a *physical point forecast*, not a risk-neutral expectation.

### 🎤 Explain it to a quant in one breath
"GARCH(1,1) makes conditional variance mean-revert: ω sets the floor, α is the shock reaction, β the
persistence, and ω/(1−α−β) is the long-run variance. I'd use GJR-t in practice so the model captures
the leverage effect and fat tails — the ℙ-measure shadow of the equity skew."

---

## Section 4 — Black–Scholes and the implied-volatility *inversion*

### The intuition
Black–Scholes is a formula that turns **(spot, strike, maturity, rate, dividend, volatility)** into a
fair option price. Everything on the input side is observable **except volatility** — it's the one
unknown that captures "how uncertain is the future." 

Now flip it: the market *quotes a price*. So you can ask, "what single volatility, plugged into BS,
reproduces this exact market price?" That number is the **implied volatility (IV)**. It exists and is
unique because the BS price is **strictly increasing in volatility** (higher vol ⇒ more valuable
option, always). IV is the market's *common language*: instead of quoting messy dollar prices across
strikes, everyone quotes "vol," which normalizes things.

### The math
With continuous dividend yield $q$:
$$ d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad d_2=d_1-\sigma\sqrt T, $$
$$ C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2), \qquad P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1). $$

How to *read* this without memorizing it:
- $N(\cdot)$ is the normal CDF (a probability between 0 and 1).
- $N(d_2)$ ≈ the **risk-neutral probability the option finishes in-the-money**. So $Ke^{-rT}N(d_2)$ is
  "what you expect to pay for the stock (the strike), times the chance you'll pay it, discounted."
- $S e^{-qT} N(d_1)$ is the **present value of receiving the stock**, conditional on exercise.
- The call value is therefore *(value of the stock you'd get) − (value of the cash you'd pay)*.

**Vega** — the sensitivity of price to volatility — is the engine of the inversion:
$$ \text{Vega} = \frac{\partial C}{\partial \sigma} = S e^{-qT}\,\phi(d_1)\,\sqrt T \; > 0. $$
Vega > 0 everywhere is *why* the inverse (IV) is unique.

### What the code does
- `bs_price(...)` — vectorized Black–Scholes for calls and puts.
- `bs_vega(...)` — the vega formula.
- `implied_vol(price, ...)` — inverts BS for σ using **Brent's method** (`brentq`), a robust 1-D
  root-finder, searching σ in `[1e-4, 5]`. It first checks the quoted price isn't below intrinsic
  value (which would be arbitrage / bad data) and returns `NaN` if so.
- A **sanity check**: price a call at σ = 25%, then invert the price → recovers 0.2500. This round-trip
  proves the two functions are mutually consistent.

### 🏦 In the real world
*(Review §5, "`bs_vega` is defined but never used"; §5, "Short-maturity guard")*
- **Vega is computed but never used.** The cleaner production inversion is **Newton's method using
  vega** (fast quadratic convergence), with Brent as a fallback when Newton misbehaves in the wings.
  As written, vega is dead code that hints at a faster path the notebook doesn't take.
- **Very short maturities are numerically dangerous** — $d_1$ has $\sigma\sqrt T$ in the denominator,
  which → 0 as $T$ → 0. The notebook sidesteps this by skipping the nearest expiry; a robust tool adds
  an explicit small-$T$ guard.

### 🎤 Explain it to a quant in one breath
"BS is monotonically increasing in σ (vega > 0), so implied vol is the unique root of BS(σ) − market price = 0.
I solve it with Brent; I'd Newton-on-vega with a Brent fallback for speed and stability."

---

## Section 5 — The market implied-vol surface from the live chain

### The intuition
This is where it becomes real. You pull **every quoted AAPL option** across several expiries, clean
the quotes, invert each one to an implied vol, and assemble the cloud of points into a surface. The
craft here is the **data hygiene** — bad quotes produce a garbage surface, so most of the work is
filtering.

Two conventions make the cloud comparable across maturities:
1. **Moneyness** — instead of raw strike, use $K/S$ (how far the strike is from spot, in %). A
   90%-moneyness option means roughly the same thing on a $50 stock and a $300 stock.
2. **The forward**, $F = S e^{(r-q)T}$ — the market's expected price of the stock at expiry under ℚ.
   It's the natural center of the smile (more so than spot), because that's the no-arbitrage
   delivery price.

### Why use only OTM options?
Out-of-the-money options (cheap, all time-value, no intrinsic value) are the **most liquid and least
noisy**. And by **put–call parity**, a call and a put at the same strike and expiry must imply the
*same* volatility — so you don't need both. The convention:
- For strikes **below** the forward → use **OTM puts** (downside protection).
- For strikes **at or above** the forward → use **OTM calls** (upside).
This stitches together one clean smile from the liquid half of each side.

### What the code does
1. Estimates AAPL's **dividend yield** $q$ (with a heuristic to fix yfinance's inconsistent units).
2. Picks ~8 expiries spread across the term, skipping the nearest week (microstructure noise).
3. `build_iv_surface()`:
   - For each expiry: computes $T$, the rate $r$, and the forward $F$.
   - Takes **mid price** = ½(bid + ask); drops zero-bid, crossed (ask < bid), and sub-5-cent quotes.
   - Keeps **OTM** only (the parity logic above).
   - Inverts each surviving quote to an IV, keeping only sensible values (1%–300%).
4. Restricts to a trustworthy moneyness band (70%–130%).
5. Plots single-expiry **smile slices** (you can see the **skew**: OTM puts trade at higher IV than
   OTM calls — crash insurance is expensive), then the **3D IV surface** side-by-side with the
   **call-price surface** rebuilt from the same IVs (to show the IV surface is just the price surface
   re-expressed in "vol language").

### 🏦 In the real world
*(Review §2.2, §4 all bullets)*
- **yfinance is retail-grade and asynchronous.** Spot is "now" but option mids can be stale, and
  bid/asks can be one-sided. Every stale quote injects noise into IV. Filter on **volume / open
  interest** (not just price > 5¢) and drop stale `lastTradeDate`s.
- **Get the forward from put–call parity, not an assumed dividend yield.** The fragile
  `if q_div > 0.05: q_div /= 100` patch will silently misprice the forward the day yfinance changes
  units. The forward (and the implied dividend) fall straight out of the call−put price difference at
  each strike — use that.
- **No no-arbitrage enforcement.** The surface is interpolated with `griddata(method="linear")`, which
  (a) can leave **NaN holes outside the convex hull** of quotes and (b) does nothing to prevent
  **butterfly** (negative implied density) or **calendar** (total variance decreasing in T) arbitrage.
  A real surface must be arbitrage-free; at minimum add diagnostics (see Section 8 note).
- **Reproducibility.** Live data means the surface changes every run and can't be unit-tested.
  **Snapshot** the chain + spot + curve to a timestamped file and build from the snapshot.

### 🎤 Explain it to a quant in one breath
"I invert OTM mids — puts below the forward, calls above — because they're liquid and parity makes the
two sides share one vol. I express strikes as moneyness so maturities are comparable. In production I'd
take the forward from parity, weight by liquidity, and enforce static no-arbitrage rather than raw
linear interpolation."

---

## Section 6 — Realized vs implied: the variance risk premium

### The intuition
Lay the **ATM implied-vol term structure** (from the option market, ℚ) on top of the **GARCH
realized-vol forecast** (from history, ℙ). Almost always, **implied sits above realized**. Why? Option
*buyers* are paying for insurance against future moves, and *sellers* demand compensation for bearing
that risk. That persistent gap is the **variance risk premium (VRP)** — the reward harvested by vol
sellers. It is *not* a forecasting error; it's a risk premium, the same way equities return more than
bonds.

### What the code does
- Extracts ATM implied vol per expiry (the quote nearest moneyness 1.0).
- Overlays it on `GARCH_TERM` (from Section 3) and the trailing realized vol.
- Prints mean ATM implied vs trailing realized — implied is typically higher.

### 🏦 In the real world
*(Review §2.3 and §3, "ATM by nearest-strike", "P vs Q")*
- **Unit mismatch (a real, subtle bug).** Implied $T$ uses **/365** (calendar) while the GARCH horizon
  uses **/252** (trading). On the same "years" axis the two curves are ~4% misaligned in time. Pick one
  clock. This is exactly the unit-mixing that causes P&L attribution errors on a desk.
- **ATM should be interpolated to exactly $K=F$**, not snapped to the nearest listed strike, or the
  term structure jitters with the strike grid.
- **ℙ vs ℚ caveat.** The "gap" mixes the true premium with GARCH forecast error, because GARCH is a
  physical point forecast, not 𝔼^ℚ. Directionally right; don't oversell it as a clean premium.

### 🎤 Explain it to a quant in one breath
"Implied minus realized is the variance risk premium — sellers get paid for warehousing vol risk. The
clean way to confirm it is to short delta-hedged straddles or variance swaps and measure the harvested
premium, mindful that I'm comparing a ℚ price to a ℙ forecast."

---

## Section 7 — Implied vs the flat Black–Scholes surface

### The intuition
This section exists to make the failure of plain Black–Scholes *visual*. BS assumes **one constant σ**,
which is a perfectly **flat plane**. The notebook draws that flat plane (at the average ATM level)
underneath the real, wrinkled market surface. The market's **skew** (tilt across moneyness) and
**term structure** (drift across maturity) are precisely the features the flat plane cannot represent.
That visible gap is the entire reason the industry moved beyond constant-vol BS to the smile models in
Sections 8–9.

### What the code does
Builds a constant surface `FLAT` at the mean ATM vol and overlays it (semi-transparent grey) under the
colored market surface in 3D.

### 🎤 Explain it to a quant in one breath
"BS is a flat plane; the market is skewed and term-dependent. The whole field of local- and
stochastic-vol modeling exists to fill the gap between that plane and reality."

---

## Section 8 — SABR: the desk standard for smile interpolation

### The intuition
You have option quotes at a handful of strikes, but you need vols at **every** strike (to price a
client's exact option, or to interpolate smoothly). SABR is a small, four-parameter model whose
parameters each control one recognizable feature of a single-maturity smile:

- **α (alpha)** — the overall **level** (how high the smile sits ≈ ATM vol).
- **β (beta)** — the **backbone** (how ATM vol moves as spot moves); usually *fixed* by convention
  because it's nearly redundant with ρ.
- **ρ (rho)** — spot/vol **correlation** ⇒ the **skew/tilt**. For equities ρ is **negative** (down
  moves raise vol).
- **ν (nu)** — **vol-of-vol** ⇒ the **curvature/smile** (how much the wings lift).

### The math
SABR models the forward and its own volatility as two correlated random processes:
$$ dF = \alpha F^\beta\,dW_1,\qquad d\alpha = \nu\,\alpha\,dW_2,\qquad dW_1\,dW_2=\rho\,dt. $$
Its fame comes from **Hagan's (2002) closed-form approximation** $\sigma_{\text{SABR}}(K,F)$: instead
of simulating those SDEs, you get implied vol from an explicit formula. That makes calibrating a smile
a **fast 3-parameter least-squares fit** (α, ρ, ν, with β fixed) — fit the formula to the observed
vols by minimizing squared error.

### What the code does
- `sabr_iv(...)` — Hagan's lognormal IV approximation (with a separate ATM-limit branch where $F=K$).
- `calibrate_sabr_slice(...)` — for one expiry, least-squares fit (α, ρ, ν) with β = 0.5 fixed and
  parameter bounds.
- Loops over every expiry slice, prints the fitted parameter table, overlays SABR fits on market
  smiles, and assembles the slices into a full SABR surface.

### 🏦 In the real world — this section has the notebook's worst real problem
*(Review §2.1 — the headline finding)*

Look at the fitted parameters the notebook actually produced:

| T (years) | ρ | ν | Verdict |
|---|---|---|---|
| short tenors | ≈ −0.13 to −0.19 | large | ✅ sensible (negative skew, curved) |
| 1.51 | **+0.030** | 0.34 | ⚠️ skew sign flipped positive |
| 2.50 | **+0.999** | 0.037 | ❌ degenerate — ρ pinned on its boundary |

For an equity, ρ should be **firmly negative** at every maturity. A ρ glued to **+0.999** with ν
collapsing toward zero is the textbook signature of a **failed least-squares fit**: the long-dated
slices have few, wide, illiquid quotes, the fit is **unweighted**, and the optimizer walked off to the
constraint chasing noise. **The long end of this surface is not trustworthy.** Fixes:
- **Weight residuals by vega and/or liquidity** so noisy far-wing quotes stop dominating.
- Add a **fit-quality gate**: reject/flag a slice if RMSE is high, a parameter hits a bound, or ρ
  flips sign versus neighbors. Print per-slice RMSE so degeneracy is *visible* instead of silent.
- Penalize for **smoothness in maturity** (ρ, ν should evolve gently across T).
- Remember Hagan's formula itself can imply **negative densities at low strikes for long T** — so
  check no-arbitrage on the fitted slices.

### 🎤 Explain it to a quant in one breath
"SABR gives a smile from four economically meaningful parameters via Hagan's closed form — α sets the
level, ρ the skew, ν the curvature, β the backbone (fixed). I calibrate per slice by least squares, but
I'd weight by vega/liquidity and gate on fit quality, because unweighted fits on the illiquid long end
blow up — here ρ pinned to +0.999, which is nonsense for equities."

---

## Section 9 — Heston: stochastic volatility from the ground up

### The intuition
SABR describes one smile at a time. **Heston** is more ambitious: it makes **variance itself a random,
mean-reverting process**, and from that single dynamic it generates an *entire* surface — skew,
curvature, and the way skew flattens with maturity — out of a handful of parameters. It's the
workhorse for pricing **exotics** consistently with the vanilla surface.

### The math
$$ dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad
   dv_t = \kappa(\theta-v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad dW^S dW^v=\rho\,dt. $$
The first line is the stock (like Black–Scholes, but with random variance $v_t$). The second line is
the variance, which **mean-reverts** to a long-run level:
- $v_0$ — variance right now.
- $\theta$ — the long-run variance it reverts toward.
- $\kappa$ — the **speed** of reversion (big κ = snaps back fast).
- $\xi$ (xi) — **vol-of-vol** ⇒ controls **smile curvature**.
- $\rho$ — correlation between stock and its variance ⇒ controls **skew** (equities: ρ < 0).
- **Feller condition** $2\kappa\theta > \xi^2$ keeps variance from hitting zero.

**How it's priced.** There's no simple closed form like BS, but there's a **semi-closed form** via the
**characteristic function** (the Fourier transform of the log-price distribution). The trick (Gil–Pelaez
inversion): you can recover the in-the-money probabilities $P_1, P_2$ by integrating the characteristic
function, then
$$ C = S e^{-qT} P_1 - K e^{-rT} P_2 $$
— structurally identical to Black–Scholes, just with $P_1, P_2$ computed by integration instead of
$N(d_1), N(d_2)$. No Monte Carlo needed, which is why Heston is fast enough to calibrate to a whole
surface.

### What the code does
- `heston_cf(...)` — the characteristic function in the **"little trap"** form (Albrecher et al.).
- `heston_call(...)` — prices a European call by numerically integrating (trapezoid) to get $P_1, P_2$.
- Uses **illustrative** (not calibrated) equity-style parameters, checks the **Feller condition**
  prints true, prices a sanity ATM call, then builds a Heston IV surface by pricing on a grid and
  inverting each price back to BS implied vol.
- The final markdown reads the surface: even with constant parameters, ρ < 0 creates the downside
  skew, ξ creates curvature, and the skew flattens with maturity — qualitatively matching the real
  AAPL surface from Section 5.

### 🏦 In the real world
*(Review §1 "Heston via the little trap…" — a genuine positive; §3, last bullet; §7 roadmap)*
- **The "little trap" form is the *right* choice — credit where due.** The naïve original-1993 Heston
  formula has a **branch-cut/discontinuity bug** in the complex logarithm that makes integration
  unstable; the little-trap form fixes it. Most homegrown implementations get this wrong, so flag it as
  a deliberate good decision, not an accident.
- **Parameters here are illustrative, not calibrated.** In production you **calibrate**
  $(\kappa,\theta,\xi,\rho,v_0)$ to the *entire* chain at once by minimizing
  $\sum(\sigma^{\text{model}}_i - \sigma^{\text{mkt}}_i)^2$ — note the contrast with SABR's *per-slice*
  fit; Heston fits the whole surface jointly.
- **Integration is fixed-truncation trapezoid** ($u \in [10^{-6}, 200]$, N=4096). Fine for a demo, but
  accuracy degrades for very short T / deep wings. **Carr–Madan FFT** or adaptive quadrature is the
  production route.

### 🎤 Explain it to a quant in one breath
"Heston makes variance a mean-reverting CIR process correlated with spot; ρ<0 gives the skew, ξ the
curvature, and the Feller condition keeps variance positive. I price via the characteristic function
and Gil–Pelaez inversion — using the little-trap form to dodge the branch-cut instability — then I'd
calibrate the five parameters jointly to the whole chain."

---

## Section 10 — Theory → practice (the takeaway map)

The notebook closes with a table mapping each concept from "the theory you knew" to "what it looks like
in practice," plus a Week-2+ wishlist. Here is the consolidated version of that map, fused with the
review so you can see, at a glance, **the gap between the learning version and the desk version**:

| Concept | What the notebook does | What a desk does (the gap) | Review ref |
|---|---|---|---|
| **Discounting** | Treasury par yields as CC zeros, linear interp | Bootstrap zeros off **OIS/SOFR**; semi-annual→CC; interp in DF/forward space | §3 |
| **Realized vol** | Close-to-close std × √252 | Add **range estimators** (Parkinson/GK/YZ) for efficiency | §3 |
| **GARCH** | GARCH(1,1), Gaussian, symmetric | **GJR/EGARCH + Student-t** for leverage & fat tails | §3 |
| **Implied vol** | Brent inversion of OTM mids | **Newton-on-vega**; small-T guard | §5 |
| **The forward** | Assumed dividend yield (fragile patch) | **Put–call parity** forward; drop the yield field | §4 |
| **Market surface** | yfinance mids, `griddata` linear | Liquidity/staleness filters; **enforce no-arbitrage**; snapshot data | §2.2, §4 |
| **Implied vs realized** | ATM-nearest, mixed 252/365 axes | Interp to **K=F**; one time convention; ℙ-vs-ℚ caveat | §2.3, §3 |
| **SABR** | Per-slice unweighted least squares | **Vega/liquidity weighting + fit gate**; long-end degenerates today | **§2.1** |
| **Heston** | Little-trap CF, illustrative params, trapezoid | Calibrate to whole chain; **FFT/adaptive** integration | §3 |

### The two changes that matter most (if you only do two)
1. **Make the surface trustworthy on the long end** — weight the SABR calibration by vega/liquidity and
   gate on fit quality. Right now ρ = +0.999 at 2.5y is silently broken.
2. **Make it reproducible and parity-based** — snapshot the data, derive the forward from put–call
   parity, and add no-arbitrage diagnostics.

Everything else (GJR-GARCH, range estimators, Newton inversion, FFT pricing) is refinement on top of
those two.

---

## Mini-glossary (for quick recall)

- **Risk-neutral measure (ℚ)** — the pricing world where assets grow at the risk-free rate and prices
  are discounted expectations. Implied vol lives here.
- **Physical measure (ℙ)** — the real world you estimate from history. Realized vol and GARCH live here.
- **Forward $F$** — the no-arbitrage expected future price, $S e^{(r-q)T}$; the natural center of a smile.
- **Moneyness** — strike relative to spot (or forward), $K/S$; makes strikes comparable across names/maturities.
- **Smile / skew** — implied vol's dependence on strike; equities **smirk** (OTM puts richer than calls).
- **Term structure** — implied (or forecast) vol's dependence on maturity.
- **Variance risk premium** — implied vol persistently above realized; the reward for selling vol.
- **Vega** — sensitivity of option price to volatility; positive everywhere, which makes IV unique.
- **Feller condition** — $2\kappa\theta > \xi^2$; keeps Heston variance strictly positive.
- **No-arbitrage (static)** — butterfly (convexity in strike ⇒ non-negative density) and calendar
  (total variance non-decreasing in maturity); a tradeable surface must satisfy both.
- **Characteristic function** — Fourier transform of the log-price density; lets you price via
  integration when no closed form exists (Heston).
