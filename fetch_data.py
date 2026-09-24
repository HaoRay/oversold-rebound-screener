import argparse
import os
import sys
import time

MAIN_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def load_universe():
    import akshare as ak
    codes = set()
    errs = []
    try:
        df = ak.stock_info_a_code_name()
        col = "code" if "code" in df.columns else df.columns[0]
        codes |= {str(x).zfill(6) for x in df[col]}
    except Exception as e:
        errs.append("stock_info_a_code_name: %r" % (e,))
    if not codes:
        try:
            df = ak.stock_info_sh_name_code()
            col = "证券代码" if "证券代码" in df.columns else df.columns[0]
            codes |= {str(x).zfill(6) for x in df[col]}
        except Exception as e:
            errs.append("stock_info_sh_name_code: %r" % (e,))
    if not codes:
        try:
            df = ak.stock_info_sz_name_code()
            col = "A股代码" if "A股代码" in df.columns else df.columns[0]
            codes |= {str(x).zfill(6) for x in df[col]}
        except Exception as e:
            errs.append("stock_info_sz_name_code: %r" % (e,))
    if not codes:
        for e in errs:
            print("  fail:", e)
    return sorted(c for c in codes if c[:3] in MAIN_PREFIXES)


def symbol(code):
    return ("sh" if code[:3] in ("600", "601", "603", "605", "688") else "sz") + code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--retry", type=int, default=3)
    a = ap.parse_args()

    import akshare as ak
    os.makedirs(a.out, exist_ok=True)
    codes = load_universe()
    print("universe: %d main-board codes" % len(codes))
    if not codes:
        print("cannot resolve universe; prepare codes manually and skip this step")
        return 2

    ok = skip = bad = 0
    failed = []
    for i, code in enumerate(codes, 1):
        dst = os.path.join(a.out, symbol(code) + ".csv")
        if os.path.exists(dst) and os.path.getsize(dst) > 200:
            skip += 1
            continue
        df = None
        for _ in range(a.retry):
            try:
                df = ak.stock_zh_a_hist_tx(symbol=symbol(code), start_date=a.start,
                                           end_date=a.end, adjust="qfq")
                break
            except Exception:
                time.sleep(1.0)
        if df is None or len(df) == 0:
            bad += 1
            failed.append(code)
            continue
        keep = [c for c in ("date", "open", "high", "low", "close", "amount") if c in df.columns]
        df[keep].to_csv(dst, index=False, encoding="utf-8")
        ok += 1
        if i % 100 == 0:
            print("  %d/%d  saved=%d skipped=%d failed=%d" % (i, len(codes), ok, skip, bad), flush=True)
        time.sleep(a.sleep)

    print("done: saved=%d skipped=%d failed=%d" % (ok, skip, bad))
    if failed:
        print("failed codes (first 20): %s" % failed[:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
