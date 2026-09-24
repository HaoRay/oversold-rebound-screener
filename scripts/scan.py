import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "..", "config.json"), encoding="utf-8"))
DATA = CFG["data_dir"]
PREFS = tuple(CFG["universe_prefixes"])
MINH = int(CFG["min_history_days"])
DISCLOSE = CFG["disclosure"]


def rsi(x, n=6):
    d = np.diff(x, prepend=x[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(float)
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(float)
    rs = np.divide(up, dn, out=np.full_like(up, np.inf), where=dn > 1e-12)
    return 100 - 100 / (1 + rs)


def list_files():
    return sorted(f for f in os.listdir(DATA) if f.endswith(".csv") and f[2:5] in PREFS)


def guess_date(files):
    if not files:
        return None
    step = max(len(files) // 30, 1)
    cand = []
    for f in files[::step]:
        try:
            d = pd.read_csv(os.path.join(DATA, f), usecols=["date"])
        except Exception:
            continue
        if len(d):
            cand.append(str(d["date"].iloc[-1]))
    return max(cand) if cand else None


def scan(target, files, limit=0):
    if limit:
        files = files[:limit]
    rows = []
    n_valid = n_up = n_down = n_flat = 0
    for k, f in enumerate(files, 1):
        try:
            df = pd.read_csv(os.path.join(DATA, f),
                             usecols=["date", "open", "high", "low", "close"])
        except Exception:
            continue
        if len(df) < MINH:
            continue
        dts = df["date"].astype(str).to_numpy()
        pos = np.where(dts == target)[0]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        if i < MINH:
            continue
        c = df["close"].to_numpy(float)
        o = df["open"].to_numpy(float)
        if not np.isfinite(c[i]) or c[i] <= 0:
            continue
        n_valid += 1
        chg = c[i] - c[i - 1]
        if chg > 0:
            n_up += 1
        elif chg < 0:
            n_down += 1
        else:
            n_flat += 1
        r6 = float(rsi(c, 6)[i])
        win = c[max(0, i - 59):i + 1]
        mx = float(np.nanmax(win)) if len(win) else np.nan
        dd = (c[i] / mx - 1) if mx > 0 else np.nan
        if r6 <= 20:
            rows.append((f[2:8], float(c[i]), r6, dd, float(o[i])))
    return rows, n_valid, n_up, n_down, n_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省=数据中最后一个交易日")
    ap.add_argument("--top", type=int, default=20, help="输出候选条数")
    ap.add_argument("--all-cand", action="store_true", help="输出全部候选")
    ap.add_argument("--limit", type=int, default=0, help="调试：只读前 N 个文件")
    a = ap.parse_args()

    if not os.path.isdir(DATA):
        print("数据目录不存在：%s" % DATA)
        return 2
    files = list_files()
    target = a.date or guess_date(files)
    yr = int(target[:4])
    th = CFG["thresholds_used"].get(str(yr)) or CFG["fixed_full_sample"]
    q1, icq = th["Q1"], th["icq2"]

    print("=" * 68)
    print("超卖反弹候选池 · 扫描报告")
    print("=" * 68)
    print("目标日期   %s%s" % (target, "（数据中最后一个交易日）" if not a.date else ""))
    print("股票池     主板 %d 个文件" % len(files))
    print("阈值       %d 年：Q1 ≤ %.4f ｜ icq2 ≥ %s 家" % (yr, q1, icq))
    print()

    rows, nv, nup, ndn, nfl = scan(target, files, a.limit)
    if nv == 0:
        print("该日期无有效样本（检查日期是否交易日）")
        return 1
    ups = nup / nv
    ice = len(rows)

    c1 = ups <= q1
    c2 = ice >= icq
    print("【信号日判定】")
    print("  C1 普跌      上涨家数占比 %.4f（%d/%d）  vs ≤ %.4f   → %s"
          % (ups, nup, nv, q1, "✓ 满足" if c1 else "✗ 不满足"))
    print("  C2 超卖密集   超卖股家数 %d（%.1f%%，RSI6≤20）  vs ≥ %s      → %s"
          % (ice, ice / nv * 100, icq, "✓ 满足" if c2 else "✗ 不满足"))
    print()
    if not (c1 and c2):
        why = []
        if not c1:
            why.append("普跌不足")
        if not c2:
            why.append("超卖不密集")
        print("  ⇒ **今日不是信号日**（%s）。约 82%% 的交易日如此，不必强求。" % "、".join(why))
    else:
        print("  ⇒ **今日是信号日**（普跌 × 超卖密集）")
        print()
        dds = np.array([r[3] for r in rows], float)
        ok = np.isfinite(dds)
        pct = np.full(len(rows), np.nan)
        s = pd.Series(dds[ok])
        pct[ok] = s.rank(pct=True).to_numpy()          # 值越小（跌得越深）百分位越低
        cand = [(rows[i], pct[i]) for i in range(len(rows)) if np.isfinite(pct[i]) and pct[i] <= 1 / 3]
        cand.sort(key=lambda t: t[0][3])               # dd60 升序 = 跌得深的在前
        print("【候选池】当日超卖股 %d 只 → dd60 底档（跌得相对深）%d 只" % (len(rows), len(cand)))
        print("  按 dd60 升序（越靠前跌得越深）：")
        print("  %-9s %9s %7s %9s %s" % ("代码", "收盘", "RSI6", "dd60", "当日分位(越低=跌得越深)"))
        show = cand if a.all_cand else cand[:a.top]
        for (code, cl, r6, dd, op), pc in show:
            print("  %-9s %9.2f %7.1f %8.2f%% %8.1f%%" % (code, cl, r6, dd * 100, pc * 100))
        if not a.all_cand and len(cand) > a.top:
            print("  ...（共 %d 只，用 --all-cand 看全部）" % len(cand))
        print()
        print("  **操作建议**：随机取 **3~10 只等权**（与全买等效），**次日开盘**买入，")
        print("               **持有 14 个交易日收盘**卖出；若 t+14 一字跌停则顺延。")
        print("               放弃 t+1 开盘 ≥ t 收盘×1.098（一字涨停）的票。")
    print()
    print("【强制披露】（输出本报告时须一并给出）")
    for i, t in enumerate(DISCLOSE, 1):
        print("  %d. %s" % (i, t))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
