import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def rsi(x, n=6):
    d = np.diff(x, prepend=x[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(float)
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(float)
    rs = np.divide(up, dn, out=np.full_like(up, np.inf), where=dn > 1e-12)
    return 100 - 100 / (1 + rs)


def rollmax(a, w, mp):
    return pd.Series(a).rolling(w, min_periods=mp).max().to_numpy(float)


def boot(d, reps=2000, seed=7):
    rng = np.random.default_rng(seed)
    k = len(d)
    if k < 5:
        return float("nan"), float("nan")
    m = np.array([d[rng.integers(0, k, k)].mean() for _ in range(reps)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--hold", type=int, default=None)
    ap.add_argument("--cost", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json"), encoding="utf-8"))
    DATA = a.data or cfg["data_dir"]
    if not os.path.isabs(DATA):
        DATA = os.path.join(here, DATA)
    PREFS = tuple(cfg["universe_prefixes"])
    MINH = int(cfg.get("min_history_days", 120))
    HOLD = int(a.hold if a.hold else cfg.get("hold_days", 14))
    COST = float(a.cost if a.cost is not None else cfg.get("cost_pct", 0.25))
    GAP = float(cfg.get("limit_gap", 1.098))

    if not os.path.isdir(DATA):
        print("data dir not found: %s" % DATA)
        print("usage: python fetch_data.py --start 20100101 --end 20260901 --out ./data")
        return 2

    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv") and f[2:5] in PREFS)
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("no csv in %s" % DATA)
        return 2
    print("files: %d   hold=%d   cost=%.2f%%" % (len(files), HOLD, COST), flush=True)

    ds = set()
    for f in files:
        try:
            ds |= set(pd.read_csv(os.path.join(DATA, f), usecols=["date"])["date"].astype(str))
        except Exception:
            continue
    dates = sorted(ds)
    didx = {d: i for i, d in enumerate(dates)}
    N = len(dates)
    print("trading days: %d  (%s .. %s)" % (N, dates[0], dates[-1]), flush=True)

    n_val = np.zeros(N, np.int64)
    n_up = np.zeros(N, np.int64)
    ns = np.zeros(N)
    nc = np.zeros(N, np.int64)
    ice_parts = []

    for k, f in enumerate(files, 1):
        try:
            df = pd.read_csv(os.path.join(DATA, f),
                             usecols=["date", "open", "high", "low", "close"])
        except Exception:
            continue
        if len(df) < MINH + HOLD + 2:
            continue
        dts = df["date"].astype(str).to_numpy()
        idx = np.array([didx.get(x, -1) for x in dts])
        if not (idx >= 0).any():
            continue
        c = df["close"].to_numpy(float)
        o = df["open"].to_numpy(float)
        m = len(c)
        r6 = rsi(c, 6)
        mx = rollmax(c, 60, 10)
        dd = np.divide(c, mx, out=np.full(m, np.nan), where=(mx > 0)) - 1.0
        chg = np.diff(c, prepend=np.nan)

        ret = np.full(m, np.nan)
        i2 = np.arange(m - HOLD)
        ent = o[i2 + 1].copy()
        ent[ent >= c[i2] * GAP] = np.nan
        ret[i2] = c[i2 + HOLD] / ent - 1.0 - COST / 100.0

        v = (idx >= 0) & (np.arange(m) >= MINH) & np.isfinite(c) & (c > 0)
        vi = idx[v].astype(np.int64)
        if len(vi):
            n_val += np.bincount(vi, minlength=N)
            upmask = (chg[v] > 0)
            if upmask.any():
                n_up += np.bincount(vi[upmask], minlength=N)

        ice = (r6 <= 20) & np.isfinite(ret) & np.isfinite(dd) & (idx >= 0)
        if ice.any():
            ice_parts.append(np.stack([idx[ice], dd[ice], ret[ice]], 1))

        nz = v & np.isfinite(ret) & ~(r6 <= 20)
        ni = idx[nz].astype(np.int64)
        if len(ni):
            ns += np.bincount(ni, weights=ret[nz], minlength=N)
            nc += np.bincount(ni, minlength=N)

        if k % 500 == 0:
            print("  scanned %d/%d" % (k, len(files)), flush=True)

    ICE = np.concatenate(ice_parts) if ice_parts else np.zeros((0, 3))
    ICE = ICE[np.argsort(ICE[:, 0], kind="stable")]
    ice_cnt = np.bincount(ICE[:, 0].astype(np.int64), minlength=N).astype(float)
    ups = np.divide(n_up, n_val, out=np.full(N, np.nan), where=n_val > 0)
    yrs = np.array([int(d[:4]) for d in dates])
    print("ice obs: %d   days: %d" % (len(ICE), N), flush=True)

    Q1 = np.full(N, np.nan)
    ICQ = np.full(N, np.nan)
    for Y in range(int(yrs.min()) + 1, int(yrs.max()) + 1):
        m = yrs < Y
        if m.sum() < 250:
            continue
        q1 = float(np.nanquantile(ups[m], 1 / 3))
        iq = float(np.nanquantile(ice_cnt[m], 2 / 3))
        Q1[yrs == Y] = q1
        ICQ[yrs == Y] = iq

    cell = np.isfinite(Q1) & np.isfinite(ups) & (ups <= Q1) & (ice_cnt >= ICQ)
    bounds = np.searchsorted(ICE[:, 0], np.arange(N + 1))
    rows = []
    for di in np.where(cell)[0]:
        lo, hi = int(bounds[di]), int(bounds[di + 1])
        if hi - lo < 3 or nc[di] < 3:
            continue
        sub = ICE[lo:hi]
        pct = pd.Series(sub[:, 1]).rank(pct=True).to_numpy()
        cand = sub[pct <= 1 / 3.0, 2]
        if len(cand) == 0:
            continue
        rows.append((int(yrs[di]), float(cand.mean()), float(ns[di] / nc[di]), int(len(cand)), int(nc[di])))

    if not rows:
        print("no signal days")
        return 1
    R = pd.DataFrame(rows, columns=["year", "strat", "ctrl", "n_cand", "n_ctrl"])
    d = (R["strat"] - R["ctrl"]).to_numpy()

    print()
    print("=" * 92)
    print("signal days: %d   total candidate trades: %d   total control trades: %d"
          % (len(R), int(R["n_cand"].sum()), int(R["n_ctrl"].sum())))
    print("=" * 92)
    lo, hi = boot(d)
    print("all:  strat %+.4f%%   control %+.4f%%   excess %+.4f%%   95%%CI[%+.4f,%+.4f]"
          % (R["strat"].mean() * 100, R["ctrl"].mean() * 100, d.mean() * 100,
             lo * 100, hi * 100))
    print()
    print("  %-6s %8s %12s %12s %12s %10s" % ("year", "days", "strat", "control", "excess", "cand/day"))
    ny = 0
    for y, g in R.groupby("year"):
        dd = (g["strat"] - g["ctrl"]).to_numpy()
        if dd.mean() > 0:
            ny += 1
        print("  %-6d %8d %+11.4f%% %+11.4f%% %+11.4f%% %10.1f"
              % (y, len(g), g["strat"].mean() * 100, g["ctrl"].mean() * 100,
                 dd.mean() * 100, g["n_cand"].mean()))
    print()
    print("years positive: %d/%d" % (ny, R["year"].nunique()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
