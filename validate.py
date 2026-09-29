# -*- coding: utf-8 -*-
"""
자동 코더(detector.py) 신뢰도 검증 — 기계 코딩 vs 사람 코딩 일치도.

논문(KCI)에서 요소누적도·대화유형·자발비율 같은 프로세스 지표를 쓰려면,
그 값을 만든 자동 코더가 사람 코딩과 얼마나 맞는지(합의도)를 먼저 보고해야 한다.
이 스크립트는 두 CSV를 (학생·차시·턴)으로 맞춰:
  · 네 맥락 요소(상황·조건·목적·대상)는 이진 분류 F1(정밀도·재현율)
  · 계기·개입·행위유형은 Cohen's κ(범주 일치, 우연 보정)
를 계산해 표로 찍는다. 표준 라이브러리만 쓴다(sklearn 불필요).

사용법:
  1) 관리자 화면에서 '사람코딩서식' CSV를 받아 두 명 이상이 각자 코드 칸을 채운다.
  2) 관리자 화면에서 '턴별코딩(기계)' CSV를 받는다.
  3)  python validate.py 기계.csv 사람.csv
      (두 코더 간 일치도만 볼 때는  python validate.py 사람A.csv 사람B.csv)

주의: 표본은 대화가 골고루 섞이도록 최소 몇십 턴 이상을 권장.
"""
import csv, sys
from collections import defaultdict

ELEMENTS = ["상황", "조건", "목적", "대상"]
CATS = ["계기", "개입", "행위유형"]
KEY = ("학생", "차시", "턴")


def _read(path):
    """CSV를 (학생,차시,턴)->row dict 로 읽는다. BOM 허용."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                k = (str(row["학생"]).strip(), str(row["차시"]).strip(), str(row["턴"]).strip())
            except KeyError:
                sys.exit("CSV에 '학생·차시·턴' 열이 있어야 해요. 관리자 화면에서 받은 서식을 쓰세요.")
            out[k] = row
    return out


def _bin(v):
    """'1'/'0'/'true'/'o'/'y'/공백 등을 0/1로. 빈칸은 None(미코딩)."""
    s = (v or "").strip().lower()
    if s == "":
        return None
    return 1 if s in ("1", "true", "t", "o", "y", "yes", "예", "있음", "참") else 0


def f1_for_element(pairs):
    """pairs: [(기계0/1, 사람0/1)] → precision/recall/f1 (사람을 정답으로 봄)."""
    tp = fp = fn = tn = 0
    for m, h in pairs:
        if m is None or h is None:
            continue
        if m == 1 and h == 1: tp += 1
        elif m == 1 and h == 0: fp += 1
        elif m == 0 and h == 1: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"n": n, "정밀도": prec, "재현율": rec, "F1": f1, "정확도": acc, "양성수(사람)": tp + fn}


def cohen_kappa(pairs):
    """pairs: [(기계라벨, 사람라벨)] 문자열 범주 → Cohen's κ + 단순일치도."""
    pairs = [(a, b) for a, b in pairs if a not in (None, "") and b not in (None, "")]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "일치도": 0.0, "kappa": None}
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    labels = set(a for a, _ in pairs) | set(b for _, b in pairs)
    ma = defaultdict(int); mb = defaultdict(int)
    for a, b in pairs:
        ma[a] += 1; mb[b] += 1
    pe = sum((ma[l] / n) * (mb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    return {"n": n, "일치도": po, "kappa": kappa}


def _interpret(k):
    if k is None: return "-"
    if k < 0.20: return "매우 낮음"
    if k < 0.40: return "낮음"
    if k < 0.60: return "보통"
    if k < 0.80: return "상당함(양호)"
    return "거의 완전"


def main():
    if len(sys.argv) != 3:
        sys.exit("사용법: python validate.py <기계코딩.csv> <사람코딩.csv>")
    A, B = _read(sys.argv[1]), _read(sys.argv[2])
    keys = sorted(set(A) & set(B))
    if not keys:
        sys.exit("두 파일에서 겹치는 (학생·차시·턴)이 없어요. 같은 표본을 코딩했는지 확인하세요.")
    only = (set(A) ^ set(B))
    print("=" * 64)
    print(f"  자동 코더 신뢰도 — 대조 턴 {len(keys)}개"
          + (f"  (한쪽에만 있는 턴 {len(only)}개는 제외)" if only else ""))
    print("=" * 64)

    print("\n[ 네 맥락 요소 — 이진 분류 F1 (사람 코딩을 정답으로) ]")
    print(f"  {'요소':<6}{'n':>5}{'정밀도':>9}{'재현율':>9}{'F1':>8}{'정확도':>9}{'양성':>6}")
    f1s = []
    for e in ELEMENTS:
        pairs = [(_bin(A[k].get(e)), _bin(B[k].get(e))) for k in keys]
        m = f1_for_element(pairs); f1s.append(m["F1"])
        print(f"  {e:<6}{m['n']:>5}{m['정밀도']:>9.2f}{m['재현율']:>9.2f}"
              f"{m['F1']:>8.2f}{m['정확도']:>9.2f}{m['양성수(사람)']:>6}")
    if f1s:
        print(f"  {'평균':<6}{'':>5}{'':>9}{'':>9}{sum(f1s)/len(f1s):>8.2f}")

    print("\n[ 범주 코드 — Cohen's κ (우연 보정 일치도) ]")
    print(f"  {'항목':<8}{'n':>5}{'단순일치':>10}{'kappa':>9}   해석")
    for c in CATS:
        pairs = [(A[k].get(c), B[k].get(c)) for k in keys]
        m = cohen_kappa(pairs)
        ks = "-" if m["kappa"] is None else f"{m['kappa']:>9.2f}"
        po = f"{m['일치도']:>10.2f}" if m["n"] else f"{'-':>10}"
        print(f"  {c:<8}{m['n']:>5}{po}{ks}   {_interpret(m['kappa'])}")
    print("\n  ※ κ 0.6↑면 대체로 사용 가능, 0.8↑면 우수. 낮으면 해당 코드 규칙을 고쳐 재코딩.")
    print("=" * 64)


if __name__ == "__main__":
    main()
