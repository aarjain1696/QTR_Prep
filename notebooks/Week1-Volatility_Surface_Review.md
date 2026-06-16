# Code Review — `Week1-Volatility_Surface.ipynb`

*Reviewer's lens: a quant on a derivatives / vol-trading desk reading this as if it were a junior's
first cut of a surface-building tool. Graded on correctness, methodology, data hygiene, and "would
this survive contact with a trading book."*

**Verdict:** Excellent as a *learning artifact* — the pedagogy, narrative, and breadth (curve →
realized → GARCH → IV → SABR → Heston) are genuinely strong and mostly methodologically sound. It is
**not** desk-grade as a tool, and shouldn't pretend to be: it relies on retail data (yfinance),
skips no-arbitrage enforcement, and the SABR calibration is silently producing degenerate fits on the
long-dated slices. Below: what's good, what's wrong, and what I'd change, ordered by how much it
matters.

---

## 1. What's done well (keep this)

- **Conceptual spine is correct.** The three-way split of *realized* (P-measure, backward),
  *implied* (Q-measure, forward), *model* vol is stated up front and respected throughout. Most
  people conflate these for years.
- **Forward-based, dividend-aware IV inversion.** Using `F = S·e^(r−q)T`, selecting **OTM** options
  (puts below F, calls above F), and inverting on mids is exactly the right hygiene. Many homegrown
  tools wrongly invert ITM options or ignore dividends.
- **Heston via the "little trap" characteristic function** (Albrecher et al.) is the *correct* stable
  parameterization — it avoids the branch-cut/discontinuity bug that plagues naïve implementations of
  the original 1993 Heston formula. Good instinct.
- **Sanity checks are present and meaningful**: the BS→IV round-trip (recovers 0.25) and the explicit
  Feller-condition print. This is the right reflex.
- **Honest footnotes.** The notebook repeatedly flags its own simplifications (par-yield-as-zero,
  assumed dividend yield vs put–call parity, illustrative Heston params). Intellectual honesty that a
  reviewer respects.
- **Interactive 3D (plotly).** Correct tool choice — the *shape* of the surface is the entire point,
  and a static plot hides it.

---

## 2. Bugs / red flags (fix before trusting any number)

### 2.1 SABR calibration is silently degenerating on long-dated slices — **most important issue**
Look at the fitted parameter table:

| T (y) | ρ | ν |
|---|---|---|
| 1.51 | **+0.030** | 0.338 |
| 2.50 | **+0.999** | 0.037 |

For **equity** smiles ρ should be solidly **negative** (downside skew → spot/vol anti-correlation).
A ρ pinned at **+0.999 on the boundary** with ν collapsing to ~0.04 is the classic signature of a
**failed least-squares fit** — the optimizer ran out of well-conditioned data and walked to the
constraint. Cause: the long-dated slices have few, wide, illiquid quotes, and the fit is
**unweighted**, so a handful of noisy far-wing points dominate. A desk would never ship this; the
long end of the surface is effectively garbage.

**Fix:**
- **Weight residuals by vega and/or liquidity** (bid-ask tightness, open interest). Equal-weighting
  vol points over-fits the illiquid wings.
- Add a **fit-quality gate**: reject/flag a slice if RMSE > threshold, if a parameter hits a bound,
  or if ρ flips sign versus neighboring maturities.
- Consider calibrating ρ and ν with **maturity smoothness penalties** (they should evolve smoothly in
  T), or fit the term structure jointly rather than fully independently per slice.
- Sanity-print the per-slice residual RMSE alongside the params so degeneracy is *visible*.

### 2.2 No no-arbitrage enforcement anywhere
The IV surface is built by `griddata(method="linear")` over scattered points, and SABR slices are
stitched independently. Nothing checks:
- **Butterfly (strike-convexity)** arbitrage → implied risk-neutral density can go negative.
- **Calendar (total-variance monotonicity in T)** arbitrage.

Hagan's SABR is itself known to admit **negative densities at low strikes for long T**. For a learning
notebook this is acceptable *if labeled*, but it's currently only mentioned as "Week-2 work." I'd
promote at least a **diagnostic**: compute `w(k,T)=σ²T` and check ∂w/∂T ≥ 0 (calendar) and the
Durrleman/butterfly condition on each slice. Cheap to add, and it teaches the single most important
property of a *tradeable* surface.

### 2.3 Two different "year" conventions sneak into the same plot
- Option maturity: `T = days/365` (calendar) — correct for discounting/expiry.
- GARCH term structure x-axis: `horizon_years = days/252` (trading).

In **Section 6** the ATM-implied term structure (365-based) is overlaid against the GARCH forecast
(252-based) on a shared "maturity (years)" axis. They're ~4% apart in horizon units, so the curves are
silently misaligned in T. Not catastrophic, but it's exactly the kind of unit-mixing that causes real
P&L attribution errors. Pick one clock for any chart that overlays two term structures (here: convert
the GARCH horizon to calendar years, or annualize realized vol on a 365 basis for this comparison).

---

## 3. Methodology limitations (correct for a learning notebook, wrong for a desk)

- **Par yields treated as zero/CC rates.** `r_cc = ln(1+y)` assumes annual compounding, but Treasury
  CMT yields are **bond-equivalent (semi-annual)**; the cleaner map is `r_cc = 2·ln(1+y/2)`
  (~4 bp difference at 4%, confirmed numerically). Bigger picture: a desk **bootstraps** zeros from
  OIS/SOFR and extracts the **option-implied forward via put–call parity**, not from an assumed
  dividend yield. The notebook already concedes this — good — but the forward error feeds directly into
  the moneyness axis and the OTM/ITM split.
- **Linear interpolation of the rate curve.** Linear-in-zero-rate can produce kinky/negative *forward*
  rates. Desks interpolate in log-discount-factor or forward space. Minor at this horizon.
- **GARCH(1,1) with Gaussian innovations and no leverage term.** Two issues for *equities*
  specifically:
  1. Returns are fat-tailed → use **Student-t** innovations (`dist="t"`).
  2. The **leverage effect** (vol reacts more to down moves) is the time-series cousin of the very
     skew the notebook spends Section 5 admiring — yet plain GARCH(1,1) is symmetric and can't see it.
     **GJR-GARCH or EGARCH** would close that loop beautifully and is a one-line change in `arch`.
- **Close-to-close realized vol only.** Fine, but noisy. Mentioning/adding a range-based estimator
  (Parkinson / Garman-Klass / Yang-Zhang) using the OHLC you already pulled would cut estimator
  variance materially — a natural teaching point on *estimation* vs *modeling* of vol.
- **P vs Q comparison (Section 6).** Overlaying a GARCH **physical-measure** forecast against
  **risk-neutral** implied and calling the gap the variance risk premium is directionally right and
  well-narrated — just keep flagging that GARCH is a *point forecast under P*, not 𝔼^Q, so the "gap"
  mixes a genuine premium with forecast error.
- **ATM by nearest-strike.** ATM implied is taken as the quote nearest moneyness 1.0; cleaner to
  **interpolate to exactly ATM-forward** (K=F) per slice so the term structure isn't jittered by the
  strike grid.
- **Heston integration is fixed-truncation trapezoid** (`u∈[1e-6,200]`, N=4096). Fine for illustration,
  but accuracy degrades for very short T / deep wings. Carr–Madan FFT or an adaptive quadrature is the
  production route.

---

## 4. Data & reproducibility (the quiet desk-killer)

- **yfinance option chains are retail-grade and asynchronous.** Quotes can be stale, the spot `S0` is
  "now" while option mids may be delayed, and bid/ask can be locked or one-sided. This injects noise
  straight into every IV. At minimum, filter on **volume / open interest** (not just `mid>0.05`) and
  consider dropping quotes whose `lastTradeDate` is stale.
- **The dividend-yield patch is fragile.** The `if q_div > 0.05: q_div /= 100` heuristic papers over
  yfinance's inconsistent units. It works today but will silently misprice forwards the day yfinance
  changes format again. Prefer extracting the forward from **put–call parity** and sidestepping the
  field entirely.
- **Non-reproducible by construction.** Every run hits live data, so the surface (and this review's
  numbers) change daily and results can't be reproduced or unit-tested. **Snapshot the raw chain + spot
  + curve to a timestamped file** (parquet/CSV) and let the notebook run from the snapshot. This is the
  single highest-leverage change for it to behave like real research code.
- **`griddata` linear interpolation returns NaN outside the convex hull** of the scattered points →
  holes/edges on the rendered surface. Expected, but worth a note or a fill strategy.

---

## 5. Engineering / code-quality nits

- **`warnings.filterwarnings("ignore")` globally** suppresses *everything*, including numerical
  warnings (overflow in the Heston CF, optimizer non-convergence, RuntimeWarnings in `np.log`). Scope
  it to the specific noisy calls, or at least re-enable around the calibration so failures are visible.
- **`bs_vega` is defined but never used** — `implied_vol` uses Brent, not Newton. Either wire vega into
  a Newton step (with Brent fallback) or drop it; right now it's dead code that implies a code path that
  doesn't exist.
- **Silent `except: continue`** around chain pulls swallows data errors. For a tool you'd at least
  count/log skipped expiries so a half-empty surface doesn't look "fine."
- **`iterrows()` in `build_iv_surface`** is slow and un-idiomatic; the BS/IV code is already vectorized,
  so the inversion could be done array-wise per expiry. Negligible at this size, but it's the wrong
  habit to bake in.
- **Short-maturity guard for IV.** `bs_price`/`vega` divide by `σ√T`; very short expiries can blow up
  `d1`. The notebook mitigates by skipping the nearest expiry (good), but an explicit small-T guard
  would be more robust.

---

## 6. What I would change — prioritized

**Tier 1 — correctness / trust**
1. **Weight the SABR fit** (vega/liquidity), add a **fit-quality gate**, and print per-slice RMSE. Kill
   the ρ=+0.999 degeneracy.
2. Add **no-arbitrage diagnostics** (calendar monotonicity of total variance + butterfly/convexity).
3. **Fix the 252-vs-365 axis mismatch** in the implied-vs-realized overlay.

**Tier 2 — realism**
4. **Snapshot data** to disk and run from the snapshot → reproducible.
5. Extract the **forward from put–call parity** instead of the dividend-yield field.
6. Switch GARCH to **GJR/EGARCH + Student-t** to capture leverage and fat tails (ties directly to the
   skew story).

**Tier 3 — polish**
7. Filter chains on **liquidity (OI/volume)**; handle stale quotes.
8. Use the correct **semi-annual→CC** rate conversion; interpolate the curve in DF/forward space.
9. Scope the warning suppression; remove or wire in `bs_vega`; vectorize the IV inversion.
10. Interpolate ATM to **exactly K=F** for the term structure.

---

## 7. Suggested Week-2+ roadmap (extends the notebook's own outlook)

- **Calibrate Heston to the whole chain** (it's currently illustrative) and compare its fitted surface
  to SABR's stitched slices — a great lesson in global vs local fitting.
- **Add Dupire local vol** (from the smoothed, arbitrage-free call surface) and contrast local vs
  stochastic vol dynamics (sticky-strike vs sticky-delta behavior).
- **Greeks off the surface**: vega, and the smile Greeks **vanna/volga** — the actual reason a desk
  builds the surface in the first place.
- **Backtest the variance risk premium**: systematically short variance (delta-hedged straddles or
  variance swaps) and measure the realized premium and its drawdowns — turns Section 6 from a picture
  into a strategy.

---

### Bottom line
As a Week-1 learning notebook this is well above average: correct concepts, the right *stable* Heston
form, honest caveats, and a clean theory→practice arc. The two things that separate it from real desk
code are **arbitrage-aware, liquidity-weighted calibration** (the SABR long-end is currently broken)
and **reproducible, parity-derived data**. Fix those two and it stops being a demo and starts being a
tool.
