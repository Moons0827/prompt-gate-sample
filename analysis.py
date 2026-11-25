"""본실험 사전 등록 분석(PREREGISTRATION.md 4절)을 그대로 돌리는 스크립트.

서버가 아니라 연구자 컴퓨터에서 돌린다. 필요한 것: python 3.10+, numpy, pandas, scipy.

    python analysis.py 이중코딩_최종코드.csv [턴별코딩.csv] [--out 결과.csv]

입력
  최종코드 CSV  관리자 화면 '최종 코드 CSV'(A=B면 그 값, 다르면 C 합의 값)
  턴별 CSV      (선택) '턴별 코딩 CSV' 전체 차시 — 조작 점검(AI 되묻기 비율)에 쓴다

분석 규칙(사전 등록과 같음)
  결과 변수   한 문항의 맥락 요소 수(상황·조건·목적·대상, 0~4)를 학생별로 그 회차 문항 평균
  공변량      사전 회차 평균(없으면 전체 평균으로 채우고 결측 표시 변수를 넣음), 학급 고정효과
  H1·H2       단계적 감소 네 칸: 사후 ~ 점검요청 + 표시등 + 상호작용(±0.5 부호화) + 공변량
  H3          '표시등 켬·점검요청 없음·단계적 감소' 칸과 '계속 되묻기' 칸: 사후 ~ 계속되묻기 + 공변량
  검정        HC3 강건 표준오차, 양측, 세 가설 Holm 보정, α=.05
  효과크기    계수 ÷ 분석 표본 결과 변수의 표준편차(d)
"""
import sys, argparse
import numpy as np
import pandas as pd
from scipy import stats

EL = ["상황", "조건", "목적", "대상"]


def ols_hc3(y, X, names):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / np.clip(1 - h, 1e-8, None)) ** 2
    V = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.diag(V)); df = n - k
    t = b / se; p = 2 * stats.t.sf(np.abs(t), df)
    return {nm: {"b": b[i], "se": se[i], "t": t[i], "p": p[i], "df": df} for i, nm in enumerate(names)}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def load_final(path):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    for e in EL:
        d[f"최종_{e}"] = pd.to_numeric(d[f"최종_{e}"], errors="coerce")
    left = d[[f"최종_{e}" for e in EL]].isna().any(axis=1).sum()
    if left:
        print(f"[주의] 합의가 남은 행 {left}개는 분석에서 뺍니다. 먼저 불일치를 합의해 주세요.")
    d = d.dropna(subset=[f"최종_{e}" for e in EL]).copy()
    d["요소수"] = d[[f"최종_{e}" for e in EL]].sum(axis=1)
    d["표시등"] = pd.to_numeric(d["표시등"], errors="coerce")
    d["점검요청"] = pd.to_numeric(d["점검요청"], errors="coerce")
    d["계속"] = (d["지원방식"] == "계속되묻기").astype(int)
    return d


def wave_scores(d):
    t = d[d["종류"] == "전이 과제"]
    s = t.groupby(["학생", "전이회차"])["요소수"].mean().unstack()
    info = d.drop_duplicates("학생").set_index("학생")[["학급", "표시등", "점검요청", "계속"]]
    return info.join(s, how="left")


def design(df, outcome, cov, terms):
    df = df.dropna(subset=[outcome]).copy()
    cols, names = [np.ones(len(df))], ["절편"]
    for nm, v in terms(df):
        cols.append(v); names.append(nm)
    if cov is not None:
        c = df[cov]; miss = c.isna().astype(float)
        cols.append(c.fillna(c.mean()).values); names.append("사전")
        if miss.sum():
            cols.append(miss.values); names.append("사전결측")
    for cl in sorted(df["학급"].unique())[1:]:
        cols.append((df["학급"] == cl).astype(float).values); names.append(f"학급[{cl}]")
    return df, np.column_stack(cols), names


def run_hyp(sc, outcome, cov="사전"):
    res = {}
    fade = sc[sc["계속"] == 0]
    df, X, nm = design(fade, outcome, cov, lambda x: [
        ("점검요청", x["점검요청"].values - .5), ("표시등", x["표시등"].values - .5),
        ("상호작용", (x["점검요청"].values - .5) * (x["표시등"].values - .5))])
    if len(df) > X.shape[1] + 2:
        f = ols_hc3(df[outcome], X, nm); sd = df[outcome].std(ddof=1)
        res["H1 점검요청"] = {**f["점검요청"], "n": len(df), "d": f["점검요청"]["b"] / sd}
        res["H2 표시등"] = {**f["표시등"], "n": len(df), "d": f["표시등"]["b"] / sd}
        res["탐색 상호작용"] = {**f["상호작용"], "n": len(df), "d": f["상호작용"]["b"] / sd}
    sub = sc[((sc["계속"] == 0) & (sc["표시등"] == 1) & (sc["점검요청"] == 0)) | (sc["계속"] == 1)]
    df, X, nm = design(sub, outcome, cov, lambda x: [("계속되묻기", x["계속"].values.astype(float))])
    if len(df) > X.shape[1] + 2 and df["계속"].nunique() == 2:
        f = ols_hc3(df[outcome], X, nm); sd = df[outcome].std(ddof=1)
        # 가설 방향은 '단계적 감소 > 계속 되묻기'이므로 부호를 뒤집어 단계적 감소 쪽 차이로 보고
        r = f["계속되묻기"]
        res["H3 단계적감소-계속되묻기"] = {**r, "b": -r["b"], "t": -r["t"], "n": len(df), "d": -r["b"] / sd}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final"); ap.add_argument("turns", nargs="?")
    ap.add_argument("--out", default="analysis_results.csv")
    a = ap.parse_args()
    d = load_final(a.final)
    sc = wave_scores(d)
    rows = []
    print(f"학생 {len(sc)}명, 조건별:", sc.groupby(["계속", "표시등", "점검요청"]).size().to_dict())

    for label, outcome in (("주 분석(사후)", "사후"), ("부 분석(지연)", "지연")):
        if outcome not in sc:
            print(f"{label}: {outcome} 회차 자료가 없어 건너뜀"); continue
        r = run_hyp(sc, outcome)
        conf = [k for k in r if k.startswith("H")]
        if label.startswith("주") and conf:
            adj = holm(np.array([r[k]["p"] for k in conf]))
            for k, p in zip(conf, adj):
                r[k]["p_holm"] = p
        print(f"\n== {label} ==")
        for k, v in r.items():
            print(f"{k:22s} n={v['n']:4d}  b={v['b']:+.3f}  SE={v['se']:.3f}  t({v['df']})={v['t']:+.2f}"
                  f"  p={v['p']:.4f}" + (f"  Holm p={v['p_holm']:.4f}" if "p_holm" in v else "") + f"  d={v['d']:+.2f}")
            rows.append({"분석": label, "검정": k, **{kk: v.get(kk) for kk in ("n", "b", "se", "t", "df", "p", "p_holm", "d")}})

    # 부 분석: 3·4차시 첫 발화(지원이 있는 환경), 공변량은 1차시 첫 발화
    f = d[d["종류"] == "첫 발화"].copy(); f["차시"] = pd.to_numeric(f["차시"], errors="coerce")
    if len(f):
        fs = f.groupby(["학생", "차시"])["요소수"].mean().unstack()
        base = d.drop_duplicates("학생").set_index("학생")[["학급", "표시등", "점검요청", "계속"]]
        for s in (3, 4):
            if s not in fs:
                continue
            sc2 = base.join(fs[[c for c in (1, s) if c in fs]].rename(columns={1: "사전", s: f"첫발화{s}"}))
            if "사전" not in sc2:
                sc2["사전"] = np.nan
            r = run_hyp(sc2, f"첫발화{s}")
            print(f"\n== 부 분석({s}차시 첫 발화) ==")
            for k, v in r.items():
                print(f"{k:22s} n={v['n']:4d}  b={v['b']:+.3f}  p={v['p']:.4f}  d={v['d']:+.2f}")
                rows.append({"분석": f"{s}차시 첫 발화", "검정": k, **{kk: v.get(kk) for kk in ("n", "b", "se", "t", "df", "p", "d")}})

    # 조작 점검: 3·4차시 AI 되묻기 비율이 계속 되묻기 칸에서 높아야 한다
    if a.turns:
        t = pd.read_csv(a.turns, encoding="utf-8-sig", dtype=str)
        t["AI되묻기"] = pd.to_numeric(t["AI되묻기"], errors="coerce"); t["차시"] = pd.to_numeric(t["차시"], errors="coerce")
        t = t[t["차시"].isin([3, 4])]
        g = t.groupby(["조건_지원방식", "학생"])["AI되묻기"].mean().groupby(level=0).agg(["mean", "count"])
        print("\n== 조작 점검(3·4차시 학생별 AI 되묻기 비율 평균) ==\n", g.round(3))
        for k, v in g.iterrows():
            rows.append({"분석": "조작 점검", "검정": f"되묻기 비율({k})", "n": int(v["count"]), "b": v["mean"]})

    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {a.out}")


if __name__ == "__main__":
    main()
