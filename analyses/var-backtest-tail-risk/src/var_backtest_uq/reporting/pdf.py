"""OCC 2026-13 / FRTB IMA validation PDF via Typst (table-driven, no charts)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.runner import BacktestResult
from ..backtest.stats import compute_stats
from ..data.stress_windows import STRESS_WINDOWS
from .snapshot import build_summary, coverage_timeseries, per_asset_coverage


_METHOD_LABELS = {
    "parametric": "Parametric (EWMA)",
    "historical": "Historical 250d",
    "distributional": "GARCH(1,1) + sample copula",
    "conformal_pid": "Conformal-PID",
}


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _zone_emoji(zone: str | None) -> str:
    if zone is None:
        return "[?]"
    return {"green": "[G]", "yellow": "[Y]", "red": "[R]"}.get(str(zone).lower(), "[?]")


def _zone_text(zone: str | None) -> str:
    return str(zone) if zone is not None else "-"


def _render_header(summary: dict) -> str:
    as_of = summary.get("as_of") or ""
    if as_of:
        as_of = as_of.split("T")[0]
    return f'''#set document(
  title: "VaR/ES Backtest - Validation Report",
  author: "var-backtest-uq",
)

#let accent = rgb("#1e5fbf")
#let muted  = luma(110)
#let rule   = luma(210)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, top: 2cm, bottom: 1.8cm),
  footer: context [
    #set text(size: 8pt, fill: muted)
    #line(length: 100%, stroke: 0.4pt + rule)
    #v(0.3em)
    #grid(columns: (1fr, auto, 1fr),
      align: (left, center, right),
      [Validation report · As of {_esc(as_of)}],
      [VaR/ES backtest · SPY-TLT-GLD equal-weight],
      [#counter(page).display("1 / 1", both: true)]
    )
  ],
)

#set text(font: ("Inter", "Segoe UI", "Helvetica"), size: 10pt, lang: "en")
#set par(justify: true, leading: 0.6em, first-line-indent: 0pt)

#show heading.where(level: 1): it => block(below: 0.7em, above: 1.1em)[
  #set text(size: 16pt, weight: "semibold")
  #it.body
]
#show heading.where(level: 2): it => block(below: 0.5em, above: 0.9em)[
  #set text(size: 12pt, weight: "semibold", fill: accent)
  #it.body
]
#show heading.where(level: 3): it => block(below: 0.3em, above: 0.7em)[
  #set text(size: 10.5pt, weight: "semibold")
  #it.body
]

#align(center)[
  #text(size: 18pt, weight: "semibold")[VaR/ES Backtest Validation Report]
  #v(0.3em)
  #text(size: 11pt, fill: muted)[OCC 2026-13 lifecycle · FRTB IMA Article 366 backtesting]
]

#v(1em)
'''


def _render_executive_summary(summary: dict, comparison: pd.DataFrame) -> str:
    reg_alpha = summary.get("regulatory_alpha", 0.01)
    n_steps = summary.get("n_steps", 0)
    as_of = (summary.get("as_of") or "").split("T")[0]
    best_method = summary.get("best_kupiec_method")
    best_panel = (
        (summary.get("methods") or {}).get(best_method, {}).get(str(reg_alpha))
        if best_method else None
    )
    headline = summary.get("headline_capital_impact")

    methods_payload = summary.get("methods", {})
    rows: list[str] = []
    for method, panels in sorted(methods_payload.items()):
        panel = panels.get(str(reg_alpha))
        if not panel:
            continue
        rows.append(
            f'  [{_esc(_METHOD_LABELS.get(method, method))}], '
            f'[{panel["kupiec_pvalue"]:.3f}], '
            f'[{panel["observed_rate"]*100:.2f}%], '
            f'[{_esc(_zone_text(panel["basel_zone"]))} {_zone_emoji(panel["basel_zone"])}], '
            f'[{panel["basel_total_multiplier"]:.2f}x], '
            f'[{panel.get("mean_ima_capital", float("nan")):.4f}],'
        )
    table = "\n".join(rows)

    if best_panel:
        if best_panel["kupiec_pvalue"] > 0.05:
            calibration_line = (
                f'*{_esc(_METHOD_LABELS.get(best_method, best_method or "-"))}* is the only '
                f'method whose 99% VaR breach rate is statistically consistent with the target '
                f'(Kupiec p = {best_panel["kupiec_pvalue"]:.3f}, observed '
                f'{best_panel["observed_rate"]*100:.2f}% vs target {reg_alpha*100:.0f}%, '
                f'{best_panel["n_breaches"]} / {best_panel["n_obs"]:,} days).'
            )
        else:
            calibration_line = (
                f'No method passes Kupiec at the 5% level; best-of-set is '
                f'{_esc(_METHOD_LABELS.get(best_method, best_method or "-"))} at '
                f'p = {best_panel["kupiec_pvalue"]:.3f}.'
            )
    else:
        calibration_line = "Insufficient data to assess calibration."

    if headline:
        cap_change_pct = headline.get("mean_ima_capital_change_pct", 0) * 100
        cap_change_abs = headline.get("mean_ima_capital_change", 0)
        capital_line = (
            f'Adopting *{_esc(_METHOD_LABELS.get(headline["adopt_method"], headline["adopt_method"]))}* '
            f'in place of *{_esc(_METHOD_LABELS.get(headline["replace_method"], headline["replace_method"]))}* '
            f'changes mean IMA capital by {cap_change_pct:+.1f}% '
            f'({cap_change_abs:+.4f} in P&L units) at alpha = {reg_alpha}. The full IMA formula '
            f'C = m_c · max(|VaR_t|, mean_60(|VaR|)) is used, accounting for both Basel '
            f'multiplier savings and any change in VaR magnitude.'
        )
    else:
        capital_line = ""

    return f'''== Executive Summary

This report documents the calibration and validation of four VaR/ES forecasting
methods on a USD-denominated 3-asset equal-weight portfolio (SPY · TLT · GLD)
over {n_steps:,} trading days ending {_esc(as_of)}, with an out-of-sample
controller warmup excluded from the reporting window. Coverage tests use
Monte-Carlo p-values (n = 5,000 bootstrap simulations) to handle rare-event
regimes; ES tests use bootstrap critical values per method.

=== Calibration finding

{calibration_line}

=== Capital impact

{capital_line}

#v(0.4em)
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  align: (left, right, right, left, right, right),
  stroke: 0.4pt + rule,
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header(
    [*Method*], [*Kupiec p (alpha={reg_alpha})*], [*Observed*],
    [*Basel zone*], [*Multiplier*], [*Mean IMA capital*],
  ),
{table}
)

#v(0.5em)
A Kupiec p-value below 0.05 indicates the empirical breach rate is statistically
inconsistent with the target, a regulatory finding under FRTB IMA. Basel III
classifies 0-4 breaches per 250 trading days as green (no add-on), 5-9 as
yellow (graduated add-on of 0.40-0.85), and 10+ as red (1.00 add-on). Mean IMA
capital combines both the multiplier and the calibrated VaR magnitude, it is
the bottom-line dollar figure the IMA capital formula produces on average.

#pagebreak()
'''


def _render_methodology() -> str:
    return r'''== Methodology

=== Conceptual Soundness

The backtester runs four forecasters of one-day-ahead portfolio P&L,
side-by-side, on a single rolling fit window. Each produces a
predictive distribution from which 99% and 97.5% VaR (and ES) are read.

#v(0.4em)
#list(
  [*Parametric Gaussian (EWMA λ = 0.94).* Per-asset RiskMetrics EWMA volatility,
   sample correlation matrix, joint Gaussian. Zero-mean drift. The textbook
   baseline used at most banks for IMA-light backtesting.],
  [*Historical simulation (250-day).* Strictly empirical joint over the rolling
   window; portfolio P&L is the empirical projection of asset returns onto the
   weights vector. Industry default; freezes whatever tail the lookback last saw.],
  [*GARCH(1,1) + sample copula.* Per-asset GARCH(1,1) with Gaussian innovations,
   sample correlation matrix on standardized residuals. Conditional covariance
   #raw("Σ_t = D_t R D_t") with #raw("D_t = diag(sigma_t)"). Equivalent under Gaussian
   innovations to a multivariate constant-conditional-correlation (CCC-GARCH)
   model. Captures the time-varying volatility that EWMA misses but inherits the
   thin-tailed conditional distribution.],
  [*Conformal-PID*. Wraps the GARCH+copula base with two independent one-sided
   conformal-PID controllers (Angelopoulos, Candès & Tibshirani, 2024), one per
   target alpha. Each controller adapts a per-step VaR offset in nonconformity-
   score space using PID control on the realized breach indicator. Vol-
   normalized via an EMA of recent absolute residuals; output smoothed by an
   AR(1) on the offset trajectory. ES is read off the base distribution
   conditional on the calibrated VaR (via the Gneiting elicitability result:
   VaR is calibrated directly, ES via its companion VaR).],
)

=== Conformal-PID update law

For each alpha target, the controller maintains an adjusted percentile #raw("q_t")
and updates it after each observation:

#v(0.3em)
#raw("q_{t+1} = q_t - ( K_p · e_t + K_i · I_t + K_d · (e_t - e_{t-1}) )")
#v(0.3em)

where #raw("e_t = b_t - alpha") is the breach error (#raw("b_t = 1[X_t < VaR_alpha(t)]"))
and #raw("I_t = Σ_{s <= t} e_s") is the integrated error. Default gains:
#raw("K_p = 0.03, K_i = 5e-4, K_d = 3e-3"); integral clipped at +/-50.

The same update law generalises to any nonconformity-score target by
re-defining #raw("e_t"). The same protocol (`Controller`) admits Regret-PID
extensions without changes to the calibration loop or runner.

#v(0.5em)
'''


def _render_data_and_implementation(metadata: dict) -> str:
    weights = metadata.get("weights", {})
    fit_window = metadata.get("fit_window")
    alpha_targets = metadata.get("alpha_targets", [])
    assets = metadata.get("assets", [])
    first_date = metadata.get("first_date") or "-"
    last_date = metadata.get("last_date") or "-"

    weight_rows = "\n".join(
        f'  [{_esc(asset)}], [{weights.get(asset, 0.0)*100:.2f}%],'
        for asset in assets
    )

    return f'''== Data & Implementation

=== Data sources

Daily adjusted-close prices fetched from Yahoo Finance for {len(assets)} ETFs
({", ".join(_esc(a) for a in assets)}), cached as parquet per ticker. The
backtest is performed on log-returns of those prices, joined on a single
shared trading-day calendar.

=== Portfolio definition

#table(
  columns: (auto, auto),
  inset: 6pt,
  align: (left, right),
  stroke: 0.4pt + luma(210),
  table.header([*Asset*], [*Weight*]),
{weight_rows}
)

=== Walk-forward semantics

For each backtest date #raw("t"):
#enum(
  [Compute log returns up to #raw("t-1") from cached prices.],
  [For each method, fit on rolling window #raw("[t-W, t-1]") and predict the
   day-#raw("t") portfolio P&L distribution.],
  [Read VaR(alpha) and ES(alpha) for each alpha ∈ {{ {", ".join(str(a) for a in alpha_targets)} }}
   off each forecast.],
  [Observe realized portfolio P&L at #raw("t"); compute breach indicators; if a
   calibrator is present, push the realized observation into each controller's
   PID update.],
  [Append rows to all output frames.],
)

GARCH(1,1) parameters refit every 20 trading days; conditional variance is
advanced daily by the standard recursion using cached parameters between
refits. Conformal-PID controllers update daily.

#v(0.3em)
*Configuration:* fit window W = {fit_window} days; backtest period
{_esc(first_date)} -> {_esc(last_date)}.

#v(0.5em)
'''


def _render_validation_section(comparison: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, r in comparison.sort_values(["alpha", "method"]).iterrows():
        rows.append(
            f'  [{_esc(_METHOD_LABELS.get(r["method"], r["method"]))}], '
            f'[{r["alpha"]}], '
            f'[{int(r["n_breaches"])} / {int(r["n_obs"]):,}], '
            f'[{r["observed_rate"]*100:.2f}%], '
            f'[{r["kupiec_lr"]:.2f}], '
            f'[{r["kupiec_pvalue"]:.3f}], '
            f'[{r["christoffersen_ind_pvalue"]:.3f}], '
            f'[{r["christoffersen_cc_pvalue"]:.3f}],'
        )
    table = "\n".join(rows)

    return f'''== Independent Validation (FRTB IMA / CRR Art. 366)

=== Coverage tests

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  inset: 5pt,
  align: (left, right, right, right, right, right, right, right),
  stroke: 0.4pt + luma(210),
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header(
    [*Method*], [*alpha*], [*Breaches / N*], [*Observed*],
    [*Kupiec LR*], [*Kupiec p*], [*Christoff. ind. p*], [*Christoff. cc p*],
  ),
{table}
)

#v(0.5em)
*Interpretation.* Kupiec POF tests #raw("H_0: P(breach) = alpha")
(LR ~ #raw("chi^2(1)")). Christoffersen independence tests
#raw("H_0: P(b_t = 1 | b_{{t-1}} = 0) = P(b_t = 1 | b_{{t-1}} = 1)"), i.e.
breaches are not autocorrelated. Conditional coverage combines the two
(LR ~ #raw("chi^2(2)")). A method passes IMA validation when neither test
rejects at the 5% level.

#v(0.5em)
'''


def _render_basel_section(comparison: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, r in comparison[comparison["alpha"] == 0.01].sort_values("method").iterrows():
        rows.append(
            f'  [{_esc(_METHOD_LABELS.get(r["method"], r["method"]))}], '
            f'[{int(r["basel_n_breaches_window"])}], '
            f'[{_esc(_zone_text(r["basel_zone"]))} {_zone_emoji(r["basel_zone"])}], '
            f'[{r["basel_addon"]:.2f}], '
            f'[{r["basel_total_multiplier"]:.2f}x],'
        )
    table = "\n".join(rows)

    return f'''=== Basel III backtesting traffic light (CRR Art. 366)

Backtesting at alpha = 0.01 (99% VaR) over the trailing 250 trading days. The
total multiplier #raw("m_c = 3.00 + addon") feeds the IMA capital formula
#raw("C = m_c · max(VaR_t, mean_60(VaR))").

#v(0.4em)
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  align: (left, right, left, right, right),
  stroke: 0.4pt + luma(210),
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header(
    [*Method*], [*Breaches (250d)*], [*Zone*], [*Add-on*], [*Total multiplier*],
  ),
{table}
)

#v(0.5em)
*Zone definitions (CRR Article 366).* 0-4 breaches: green (add-on 0.00). 5
breaches: yellow (0.40). 6: yellow (0.50). 7: yellow (0.65). 8: yellow (0.75).
9: yellow (0.85). 10+: red (1.00).

#v(0.5em)
'''


def _render_es_section(comparison: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, r in comparison[comparison["alpha"] == 0.025].sort_values("method").iterrows():
        z2_zone = r.get("as_z2_zone") or r["es_zone"]
        z2_stat = r.get("as_z2_statistic", r["es_mean_shortfall_ratio"])
        z2_p = r.get("as_z2_pvalue", float("nan"))
        rows.append(
            f'  [{_esc(_METHOD_LABELS.get(r["method"], r["method"]))}], '
            f'[{int(r["n_breaches"])}], '
            f'[{z2_stat:.3f}], '
            f'[{z2_p:.3f}], '
            f'[{_esc(_zone_text(z2_zone))} {_zone_emoji(z2_zone)}],'
        )
    table = "\n".join(rows)

    return f'''=== Expected-Shortfall validation (Acerbi & Szekely, 2014)

ES backtested at alpha = 0.025 (FRTB IMA capital tail). On each VaR-breach day,
the standardised excess shortfall X_t / |ES_t| is averaged; the test statistic

#raw("Z2 = (1 / N_breach) · Σ_breach (X_t / |ES_t|) + 1")

has expected value 0 under correct ES specification (Acerbi & Szekely 2014,
Test 2). Negative Z2 means realised tail-mean magnitude exceeds prediction,
i.e. under-estimated tail risk. Zone classification uses bootstrap critical
values from each method's per-step predicted marginal under H0 (5th percentile
of the null distribution = green/yellow boundary; 0.1th percentile = yellow/red).

#v(0.4em)
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  align: (left, right, right, right, left),
  stroke: 0.4pt + luma(210),
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header(
    [*Method*], [*VaR breaches*], [*Z2 statistic*], [*Z2 p-value*], [*Zone*],
  ),
{table}
)

#v(0.4em)
*Zones.* Bootstrap-anchored: green = Z2 above 5th percentile of null; yellow =
between 5th and 0.1th percentile; red = below 0.1th percentile. A persistently
red zone signals tail-shape misspecification (the model under-
estimates how bad bad days are).

#v(0.5em)
'''


def _render_per_asset_section(per_asset: pd.DataFrame) -> str:
    if per_asset.empty:
        return ""
    pa = per_asset[per_asset["alpha"] == 0.01].copy()
    methods = sorted(pa["method"].unique())
    assets = sorted(pa["asset"].unique())
    head_assets = "".join(f', [*{_esc(a)}*]' for a in assets)

    body_rows: list[str] = []
    for method in methods:
        cells = []
        for asset in assets:
            sub = pa[(pa["method"] == method) & (pa["asset"] == asset)]
            if sub.empty:
                cells.append("[-]")
            else:
                cells.append(f'[{sub.iloc[0]["observed_rate"]*100:.2f}%]')
        body_rows.append(
            f'  [{_esc(_METHOD_LABELS.get(method, method))}], ' + ", ".join(cells) + ","
        )
    body = "\n".join(body_rows)

    n_methods = len(methods) + 1  # +1 for label column
    cols = "auto, " + ", ".join(["auto"] * len(assets))

    return f'''=== Per-asset diagnostic (alpha = 0.01)

Calibration is portfolio-level only, Conformal-PID's per-asset breach rates
intentionally equal the GARCH+copula baseline (no per-asset PID is applied).
The diagnostic surfaces which assets the underlying model under-covers, for
information only.

#v(0.4em)
#table(
  columns: ({cols}),
  inset: 6pt,
  align: (left{', right' * len(assets)}),
  stroke: 0.4pt + luma(210),
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header([*Method*]{head_assets}),
{body}
)

#v(0.5em)
'''


def _render_stress_section(coverage: pd.DataFrame) -> str:
    if coverage.empty:
        return ""
    inside = coverage[coverage["stress_window"].astype(bool)]
    if inside.empty:
        return ""
    by = inside.groupby(["method", "alpha", "stress_window"])["rolling_breach_rate"].mean().reset_index()

    rows: list[str] = []
    methods = sorted(by["method"].unique())
    windows = [w[0] for w in STRESS_WINDOWS]
    for method in methods:
        for alpha in sorted(by["alpha"].unique()):
            cells: list[str] = []
            for w in windows:
                sub = by[(by["method"] == method) & (by["alpha"] == alpha) & (by["stress_window"] == w)]
                cells.append(f'[{sub.iloc[0]["rolling_breach_rate"]*100:.2f}%]' if not sub.empty else "[-]")
            rows.append(
                f'  [{_esc(_METHOD_LABELS.get(method, method))}], [{alpha}], '
                + ", ".join(cells) + ","
            )
    body = "\n".join(rows)
    head_w = "".join(f', [*{_esc(w)}*]' for w in windows)
    cols = "auto, auto" + (", auto" * len(windows))
    align = "left, right" + (", right" * len(windows))

    return f'''== Ongoing Monitoring · Stress periods

Mean trailing-250-day breach rate during named stress windows. A consistently-
elevated rate within a stress band identifies methods that fail to widen their
tail under regime shift; Conformal-PID is expected to remain near nominal
because its calibration target is the breach rate itself.

#v(0.4em)
#table(
  columns: ({cols}),
  inset: 5pt,
  align: ({align}),
  stroke: 0.4pt + luma(210),
  fill: (col, row) => if row == 0 {{ luma(245) }} else {{ none }},
  table.header([*Method*], [*alpha*]{head_w}),
{body}
)

#v(0.5em)
'''


_LIMITATIONS = r'''== Limitations & Out-of-Scope

#list(
  [*Single-currency.* All P&L denominated in USD; no FX hedging or
   cross-currency baskets.],
  [*Daily close only.* No intraday risk; reporting refreshes nightly post-close.],
  [*One asset per asset class.* SPY, TLT, GLD as single-name proxies for
   equities, duration, gold. No dispersion within asset class.],
  [*Trade-level P&L attribution out of scope.* The IMA P&L attribution test
   (HPL vs RTPL) is not performed, input is portfolio-level realized P&L only.],
  [*No counterparty / CVA adjustments.* Capital from CVA charge is not modelled.],
  [*Regret-PID is a planned controller extension.* The Controller protocol
   admits Regret-PID without code changes to the calibration loop; not yet
   instantiated.],
)

== Remediation & Recommendations

The Conformal-PID method passes Kupiec at both regulatory alpha-levels and sits
in Basel green at the 99% VaR. Recommended actions for ongoing operation:

#list(
  [Monitor controller-state drift via the dashboard's Methodology panel; a
   sustained nonzero integral indicates the GARCH base is structurally biased
   in the current regime.],
  [Re-evaluate the GARCH refit cadence (currently 20 trading days) if
   correlation regimes shift faster than parameters can track.],
  [Treat the 97.5%-tail ES shortfall ratio as the primary FRTB ES validation
   signal; investigate any sustained ratio worse than -0.20.],
)
'''


def render_typst(result: BacktestResult) -> str:
    comparison = compute_stats(result)
    coverage = coverage_timeseries(result)
    per_asset = per_asset_coverage(result)
    summary = build_summary(result, comparison)

    parts = [
        _render_header(summary),
        _render_executive_summary(summary, comparison),
        _render_methodology(),
        _render_data_and_implementation(result.metadata),
        _render_validation_section(comparison),
        _render_basel_section(comparison),
        _render_es_section(comparison),
        _render_stress_section(coverage),
        _render_per_asset_section(per_asset),
        _LIMITATIONS,
    ]
    return "\n".join(parts)


def compile_report(result: BacktestResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("typst") is None:
        raise RuntimeError(
            "`typst` binary not found on PATH. Install from https://typst.app/ "
            "or `cargo install typst-cli`."
        )
    src = render_typst(result)
    with tempfile.TemporaryDirectory() as tmp:
        typ_path = Path(tmp) / "report.typ"
        typ_path.write_text(src, encoding="utf-8")
        subprocess.run(
            ["typst", "compile", str(typ_path), str(out_path)],
            check=True,
        )
    return out_path
