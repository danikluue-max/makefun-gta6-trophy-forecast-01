"""GTA VI trophy rarity forecast: cohort-accounting forecaster.

The idea in one paragraph. Exophase's page only changes when it refreshes,
and one refresh does two things at once: it adds newly tracked players (who
have earned almost nothing yet) and it records the progress of the players
already tracked. So a trophy's earn rate is a ratio of two counts -- earners
over players tracked -- and this program works in counts, not rates. It reads
every refresh step in the case's history, measures how many earners each step
added per newly tracked player (the cohort earn rate), fits how the tracked
player count itself grows, and rolls both forward to the target: future
earners = today's earners + future new players x cohort earn rate, future
rate = future earners / future players. When the history is flat (no refresh
seen yet) it falls back on a launch-night prior: one refresh is due, players
multiply, and the new cohort holds the one trophy everybody gets plus a small
share of every other trophy's current rate. A log-time trend line is blended
in when the history has real movement, and the spread comes from how much the
refresh steps disagree with each other.

Standard library only, no network, deterministic: the same case always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORM = NormalDist()

# --- launch-night prior, used only when the history shows no refresh -------
# Exophase's first refresh (rules): 638 -> 3,868 players tracked, 943 ->
# 4,276 trophies earned. The one trophy at 100% accounts for 3,230 of the
# 3,333 new earns; the other 103 earns spread over the remaining trophies
# are ~7% of what those trophies' current rates would give at full strength.
PRIOR_GROWTH = 6.0        # players tracked multiply by ~6 at a refresh
PRIOR_BETA = 0.07         # new cohort earns ~7% of a trophy's current rate
PRIOR_SPREAD = 0.20       # logit sigma when no refresh has been observed
FLOOR_SPREAD = 0.15       # logit sigma floor once steps are observed
CAP_SPREAD = 0.90
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment, release):
    return max(0.0, (moment - release).total_seconds() / 3600.0)


def least_squares(xs, ys):
    """Slope, intercept and residual std of y on x. Returns None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return slope, intercept, math.sqrt(var)


def analyse_case(case):
    """Turn one case into per-trophy (median_logit, spread) forecasts."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    release = RELEASE
    try:
        rel = case.get("game", {}).get("release_date")
        if rel:
            release = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass

    if not history:
        return [(logit_of(1.0), 0.5) for _ in trophies]

    # -- reads, oldest first, nulls carried forward per trophy --------------
    reads = []  # (moment, players, [rates])
    for h in history:
        try:
            moment = parse_time(h["at"])
            players = float(h["players_tracked"])
            rates = list(h["rates"])
        except (KeyError, TypeError, ValueError):
            continue
        if players <= 0 or len(rates) != n_tr:
            continue
        reads.append((moment, players, rates))
    if not reads:
        return [(logit_of(1.0), 0.5) for _ in trophies]

    filled = []  # (moment, players, [rates with nulls filled forward])
    last_seen = [None] * n_tr
    for moment, players, rates in reads:
        row = []
        for i, r in enumerate(rates):
            if r is None:
                r = last_seen[i]
            else:
                last_seen[i] = r
            row.append(0.01 if r is None else float(r))
        filled.append((moment, players, row))

    # Collapse consecutive identical reads: a flat stretch is one observation.
    distinct = [filled[0]]
    for row in filled[1:]:
        prev = distinct[-1]
        if row[1] != prev[1] or row[2] != prev[2]:
            distinct.append(row)
    last_moment, last_n, last_rates = distinct[-1]
    target_moment = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target_moment, release))

    # -- refresh steps: what each jump added, split into two forces --------
    # A step's new earners come from two places: players already tracked
    # progressing (scales with players x hours x share not yet earned) and
    # newly tracked players earning (scales with the new players). Pooling
    # both into one cohort rate lags reality whenever the mix shifts, so each
    # trophy gets both rates from a ridge-regularised two-parameter fit over
    # the steps; with too few steps the progress rate falls back to zero.
    steps = []  # (weight, dN, growth, dt_h, [cohort rates 0..1])
    step_rows = []  # (weight, dt_h, N_before, [rates before], [rates after])
    for a, b in zip(distinct, distinct[1:]):
        (ta, na, ra), (tb, nb, rb) = a, b
        dn = nb - na
        if dn <= 0:
            continue
        dt = max(0.25, (tb - ta).total_seconds() / 3600.0)
        cohort = []
        for i in range(n_tr):
            ea = na * ra[i] / 100.0
            eb = nb * rb[i] / 100.0
            cohort.append(min(1.0, max(0.0, (eb - ea) / dn)))
        age_h = max(0.0, (last_moment - tb).total_seconds() / 3600.0)
        weight = dn * (0.5 ** (age_h / 24.0))
        steps.append((weight, dn, nb / na, dt, cohort))
        step_rows.append((weight, dt, na, ra, rb))

    def two_component(i, c_pool):
        """(progress_per_hour, cohort_rate) for trophy i from the steps.

        Ridge-regularised least squares of each step's new earners on two
        exposures: players x hours x share not yet earned (progress) and new
        players (cohort). The ridge pulls the cohort rate toward the pooled
        one-step estimate and the progress rate toward zero when the steps
        cannot tell them apart.
        """
        if len(step_rows) < 3:
            return 0.0, c_pool
        s11 = s12 = s22 = t1 = t2 = 0.0
        for k, (w, dt, na, ra, rb) in enumerate(step_rows):
            dn = steps[k][1]
            nb = na + dn
            y = nb * rb[i] / 100.0 - na * ra[i] / 100.0
            x1 = na * dt * (1.0 - ra[i] / 100.0)
            x2 = dn
            s11 += w * x1 * x1
            s12 += w * x1 * x2
            s22 += w * x2 * x2
            t1 += w * x1 * y
            t2 += w * x2 * y
        if s11 <= 0 or s22 <= 0:
            return 0.0, c_pool
        lam1 = 0.05 * s11
        lam2 = 0.05 * s22
        a11 = s11 + lam1
        a22 = s22 + lam2
        b1 = t1
        b2 = t2 + lam2 * (c_pool if c_pool is not None else 0.0)
        det = a11 * a22 - s12 * s12
        if det <= 1e-12:
            return 0.0, c_pool
        p = (b1 * a22 - s12 * b2) / det
        c = (a11 * b2 - s12 * b1) / det
        p = min(0.2, max(0.0, p))
        c = min(1.0, max(0.0, c))
        return p, c

    # -- players tracked at the target --------------------------------------
    if steps:
        wsum = sum(s[0] for s in steps)
        g_hat = math.exp(sum(s[0] * math.log(s[2]) for s in steps) / wsum)
        interval = sum(s[0] * s[3] for s in steps) / wsum
        # Power-law fit of ln N on ln(1 + hours since release), dampened growth.
        xs = [math.log1p(hours_since_release(m, release)) for m, _, _ in distinct]
        ys = [math.log(n) for _, n, _ in distinct]
        fit_n = least_squares(xs, ys)
        if fit_n is not None and fit_n[0] > 0:
            n_target = math.exp(fit_n[1] + fit_n[0] * x_target)
        else:
            n_target = last_n * g_hat ** max(1.0, horizon_h / interval)
        # The compounding estimate is an upper bound; the power law is the
        # damped one. Keep the power law unless it looks absurd.
        n_target = min(n_target, last_n * 500.0)
        n_target = max(n_target, last_n)
    else:
        g_hat = PRIOR_GROWTH
        n_target = last_n * PRIOR_GROWTH

    # -- median by cohort accounting + progress, trend as a cross-check -----
    out = []
    for i in range(n_tr):
        earners_last = last_n * last_rates[i] / 100.0
        if steps:
            wsum = sum(s[0] for s in steps)
            c_pool = sum(s[0] * s[4][i] for s in steps) / wsum
            prog, cohort = two_component(i, c_pool)
            dn_t = n_target - last_n
            # Players already tracked keep earning (non-earners decay
            # exponentially at the progress rate); new players earn at the
            # cohort rate and then progress for half the horizon on average.
            exist_earn = (last_n - earners_last) * (1.0 - math.exp(-prog * horizon_h))
            new_earn = dn_t * (cohort + (1.0 - cohort)
                               * (1.0 - math.exp(-prog * horizon_h / 2.0)))
            earners_t = earners_last + exist_earn + new_earn
        else:
            cohort = 1.0 if last_rates[i] >= 99.99 else PRIOR_BETA * last_rates[i] / 100.0
            earners_t = earners_last + (n_target - last_n) * cohort
        rate_c = 100.0 * earners_t / n_target
        med_c = logit_of(rate_c)

        med = med_c
        s_res = 0.0
        s_disagree = 0.0
        if len(distinct) >= 3:
            xs = [math.log1p(hours_since_release(m, release))
                  for m, _, _ in distinct]
            ys = [logit_of(row[2][i]) for row in distinct]
            fit = least_squares(xs, ys)
            if fit is not None:
                med_t = fit[1] + fit[0] * x_target
                med_t = min(max(med_t, logit_of(0.01)), logit_of(99.99))
                # 0.85 two-component accounting, 0.15 the observed log-time
                # trend: the trend is a hedge for regimes where the
                # accounting breaks (growth stalls, cadence changes); the
                # accounting carries the dilution and the progress.
                med = 0.85 * med_c + 0.15 * med_t
                s_disagree = abs(med_c - med_t) / 2.0
                s_res = fit[2]

        # -- spread: steps disagreeing + trend residual + a floor -----------
        if steps and len(steps) >= 2:
            implied = []
            for s in steps:
                e = earners_last + (n_target - last_n) * s[4][i]
                implied.append(logit_of(100.0 * e / n_target))
            mean_imp = sum(implied) / len(implied)
            var_step = sum((v - mean_imp) ** 2 for v in implied) / len(implied)
            s_step = math.sqrt(var_step)
        else:
            s_step = 0.0
        if steps:
            spread = math.sqrt(FLOOR_SPREAD ** 2 + s_step ** 2
                               + (0.5 * s_res) ** 2 + s_disagree ** 2)
        else:
            spread = PRIOR_SPREAD
        spread *= 1.0 + 0.08 * math.sqrt(horizon_d)
        spread = min(1.20, max(0.12, spread))
        out.append((med, spread))
    return out


def forecast_case(case, name):
    levels = case.get("levels", LEVELS)
    zs = [NORM.inv_cdf(tau) for tau in levels]
    forecasts = analyse_case(case)
    rows = []
    for trophy, (med, spread) in zip(case["trophies"], forecasts):
        qs = [round4(inv_logit(med + z * spread)) for z in zs]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
