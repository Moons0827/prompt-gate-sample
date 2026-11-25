# -*- coding: utf-8 -*-
"""
이중 코딩(사람 두 명의 독립 코딩)과 일치도 계산
================================================

예비실험에서는 연구자 한 명이 자동 코더 결과를 보지 않고 150턴을 코딩했다.
본실험에서는 두 사람이 서로의 판정과 자동 코더 판정을 모두 모른 채 같은 시트를 코딩하고,
  · 사람 대 사람(코더 A 대 B)의 일치도로 판정 기준이 믿을 만한지,
  · 사람 대 자동 코더의 일치도로 자동 코더를 전체 자료에 써도 되는지
를 따로 확인한다.

코딩 시트(가린 것): 학생, 차시, 조건(표시등·점검 요청·지원 방식), 자동 코더 판정.
  → 코더는 어느 조건의 어느 차시 발화인지 모른다(조건을 모른 채 판정, blind coding).
  → 행 순서도 섞는다.
대응표(관리자 보관): 코딩ID ↔ 학생·차시·턴·조건·자동 판정. 일치도 계산은 서버가 이 표로 한다.

시트에 넣는 발화
  · first    : 각 학생·차시의 첫 발화 전부(주 지표)
  · turn     : 첫 발화 이후 발화 가운데 무작위 표본(기본 150개, 앞 대화를 함께 보여 줌)
  · transfer : 전이 과제 응답 전부
판정 단위: '판정할 발화'에 각 요소가 실질적으로 들어 있는가(0/1). 앞 대화는 뜻을 이해하는 데만 쓴다.
"""
from __future__ import annotations
import io, csv, random, secrets, time, threading

import store
import roster
import transfer

ELEMENTS = ("상황", "조건", "목적", "대상")
KEY_KEY = "coding_key"
CODES_KEY = "coding_codes"
KIND_LABEL = {"first": "첫 발화", "turn": "대화 중 발화", "transfer": "전이 과제"}
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# 시트 만들기
# ---------------------------------------------------------------------------
def _student_turns(transcript: list) -> list:
    return [t for t in sorted(transcript, key=lambda x: x.get("turn") or 0)
            if (t.get("student") or "").strip()]


def build(n_turns=150, seed=None, kinds=None) -> dict:
    kinds = [k for k in (kinds or ["first", "turn", "transfer"]) if k in KIND_LABEL]
    try:
        n_turns = max(0, min(2000, int(n_turns)))
    except Exception:
        n_turns = 150
    if seed is None:
        seed = secrets.randbelow(10**9)
    rng = random.Random(int(seed))
    conds = {r["학생"]: r for r in roster.export_rows()}
    items, later = [], []
    for r in store.list_all(None):
        sid, sess = r["student_id"], int(r["session"])
        if not roster.parse_sid(sid)[0]:
            continue                                  # 명단 밖(시연·연수) 접속은 넣지 않는다
        st = (r.get("state") or {}).get("transcript", [])
        ts = _student_turns(st)
        if not ts:
            continue
        for i, t in enumerate(ts):
            e = t.get("turn_elements") or {}
            base = {"sid": sid, "session": sess, "turn": t.get("turn"), "wave": "", "item": "",
                    "text": t.get("student") or "",
                    "auto": {k: int(bool(e.get(k))) for k in ELEMENTS}}
            if i == 0:
                if "first" in kinds:
                    items.append({**base, "kind": "first", "context": "(이 차시의 첫 발화)"})
            else:
                prev_ai = ts[i - 1].get("ai") or ""
                same = [p.get("student") for p in ts[max(0, i - 3):i]
                        if p.get("topic_no", 1) == t.get("topic_no", 1)]
                ctx = ""
                if same:
                    ctx += "앞 학생 발화: " + " / ".join(same)
                if prev_ai:
                    ctx += ("\n" if ctx else "") + "직전 AI 답: " + prev_ai
                later.append({**base, "kind": "turn", "context": ctx or "(앞 대화 없음)"})
    if "turn" in kinds and later:
        later.sort(key=lambda x: (x["sid"], x["session"], x["turn"] or 0))
        items += rng.sample(later, min(n_turns, len(later)))
    if "transfer" in kinds:
        for r in transfer.all_responses():
            a = r.get("auto") or {}
            items.append({"sid": r["sid"], "session": "", "turn": "", "wave": r["wave"], "item": r["item"],
                          "kind": "transfer", "text": r.get("text") or "",
                          "context": f"(전이 과제: '{r.get('title', '')}' 상황에서 AI에게 보낼 첫 질문)",
                          "auto": {k: ("" if k not in a else int(bool(a[k]))) for k in ELEMENTS}})
    if not items:
        return {"error": "코딩할 발화가 아직 없어요."}
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["id"] = f"K{i:04d}"
        c = conds.get(it["sid"], {})
        it["cond"] = {"지원방식": c.get("지원방식", ""), "표시등": c.get("표시등", ""),
                      "점검요청": c.get("점검요청", ""), "학급": c.get("학급", "")}
    with _LOCK:
        store.put_config(KEY_KEY, {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": int(seed),
                                   "n_turns": n_turns, "kinds": kinds, "rows": items})
        store.put_config(CODES_KEY, {})              # 새 시트를 만들면 이전 코딩 결과는 맞지 않으므로 지운다
    by = {k: sum(1 for x in items if x["kind"] == k) for k in KIND_LABEL}
    return {"ok": True, "seed": int(seed), "total": len(items), "by_kind": by}


def _key() -> dict:
    return store.get_config(KEY_KEY) or {}


def _codes() -> dict:
    return store.get_config(CODES_KEY) or {}


def _csv_bytes(header, rows) -> bytes:
    buf = io.StringIO(); buf.write("﻿"); w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def sheet_csv() -> bytes:
    """코더에게 줄 시트 — 학생·차시·조건·자동 판정을 모두 가린다. 빈칸에 1(있음)/0(없음)을 적는다."""
    rows = _key().get("rows", [])
    return _csv_bytes(["코딩ID", "종류", "앞 대화", "판정할 발화", *ELEMENTS, "메모"],
                      [[r["id"], KIND_LABEL[r["kind"]], r["context"], r["text"], "", "", "", "", ""]
                       for r in rows])


def key_csv() -> bytes:
    """관리자 보관용 대응표 — 코더에게 주지 않는다."""
    rows = _key().get("rows", [])
    return _csv_bytes(["코딩ID", "종류", "학생", "학급", "차시", "턴", "전이회차", "전이문항",
                       "지원방식", "표시등", "점검요청", *[f"자동_{e}" for e in ELEMENTS]],
                      [[r["id"], KIND_LABEL[r["kind"]], r["sid"], r["cond"].get("학급", ""), r["session"],
                        r["turn"], transfer.WAVE_LABEL.get(r["wave"], ""), r["item"],
                        r["cond"].get("지원방식", ""), r["cond"].get("표시등", ""), r["cond"].get("점검요청", ""),
                        *[r["auto"].get(e, "") for e in ELEMENTS]] for r in rows])


# ---------------------------------------------------------------------------
# 코딩 결과 올리기
# ---------------------------------------------------------------------------
_YES = {"1", "o", "ㅇ", "있음", "예", "y", "yes", "true", "t"}
_NO = {"0", "x", "ㄴ", "없음", "아니오", "아니요", "n", "no", "false", "f"}


def _bit(v):
    s = str(v or "").strip().lower()
    if s in _YES:
        return 1
    if s in _NO:
        return 0
    return None


def upload(coder: str, text: str) -> dict:
    coder = (coder or "").strip().upper()
    if coder not in ("A", "B", "C"):
        return {"error": "코더는 A, B, C(불일치 합의) 중에서 골라 주세요."}
    ids = {r["id"] for r in _key().get("rows", [])}
    if not ids:
        return {"error": "먼저 코딩 시트를 만들어 주세요."}
    text = (text or "").lstrip("﻿")
    try:
        rd = csv.DictReader(io.StringIO(text))
        fields = [f.strip() for f in (rd.fieldnames or [])]
    except Exception as e:
        return {"error": f"CSV를 읽지 못했어요: {e}"}
    if "코딩ID" not in fields or any(e not in fields for e in ELEMENTS):
        return {"error": "열 이름이 시트와 달라요(코딩ID, 상황, 조건, 목적, 대상이 있어야 해요)."}
    got, bad, unknown = {}, 0, 0
    for row in rd:
        row = {(k or "").strip(): v for k, v in row.items()}
        cid = (row.get("코딩ID") or "").strip()
        if not cid:
            continue
        if cid not in ids:
            unknown += 1; continue
        vals = {e: _bit(row.get(e)) for e in ELEMENTS}
        if any(v is None for v in vals.values()):
            bad += 1; continue                      # 빈칸·알 수 없는 값이 있는 행은 계산에서 뺀다
        got[cid] = vals
    if not got:
        return {"error": "읽을 수 있는 행이 없어요. 칸에 1 또는 0을 적었는지 확인해 주세요."}
    with _LOCK:
        c = _codes()
        c[coder] = got
        c.setdefault("uploaded", {})[coder] = time.strftime("%Y-%m-%d %H:%M:%S")
        store.put_config(CODES_KEY, c)
    return {"ok": True, "coder": coder, "rows": len(got), "skipped_blank": bad,
            "unknown_id": unknown, "total": len(ids)}


def clear_codes(coder: str | None = None) -> dict:
    with _LOCK:
        c = _codes()
        if coder in ("A", "B", "C"):
            c.pop(coder, None)
            c.get("uploaded", {}).pop(coder, None)
        else:
            c = {}
        store.put_config(CODES_KEY, c)
    return {"ok": True}


# ---------------------------------------------------------------------------
# 일치도
# ---------------------------------------------------------------------------
def kappa(a: list, b: list):
    """이분 판정의 Cohen's κ. 표본이 없거나 기대 일치가 1이면 None."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def f1(pred: list, gold: list):
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    if tp == 0:
        return 0.0 if (fp or fn) else None
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)


def _r(x, k=2):
    return None if x is None else round(x, k)


def report() -> dict:
    key = _key(); rows = {r["id"]: r for r in key.get("rows", [])}
    c = _codes(); A, B = c.get("A", {}), c.get("B", {})
    out = []
    for scope in ("전체", "first", "turn", "transfer"):
        ids = [i for i, r in rows.items() if scope == "전체" or r["kind"] == scope]
        if not ids:
            continue
        for e in (*ELEMENTS, "네 요소 합침"):
            els = ELEMENTS if e == "네 요소 합침" else (e,)
            ab_a, ab_b, aa, aauto, bb, bauto, cons, cauto = [], [], [], [], [], [], [], []
            for i in ids:
                auto = rows[i]["auto"]
                for el in els:
                    au = auto.get(el, "")
                    if i in A and i in B:
                        ab_a.append(A[i][el]); ab_b.append(B[i][el])
                        if A[i][el] == B[i][el] and au != "":
                            cons.append(A[i][el]); cauto.append(int(au))
                    if i in A and au != "":
                        aa.append(A[i][el]); aauto.append(int(au))
                    if i in B and au != "":
                        bb.append(B[i][el]); bauto.append(int(au))
            agree = (sum(1 for x, y in zip(ab_a, ab_b) if x == y) / len(ab_a)) if ab_a else None
            out.append({"범위": "전체" if scope == "전체" else KIND_LABEL[scope], "요소": e,
                        "A·B 함께 코딩(판정 수)": len(ab_a),
                        "A·B 일치율": _r(agree), "A·B κ": _r(kappa(ab_a, ab_b)),
                        "A·자동 κ": _r(kappa(aa, aauto)), "B·자동 κ": _r(kappa(bb, bauto)),
                        "합의본(A=B)·자동 κ": _r(kappa(cauto, cons)),
                        "합의본·자동 F1": _r(f1(cauto, cons)),
                        "합의본 '있음' 비율": _r(sum(cons) / len(cons) if cons else None)})
    return {"rows": out, "uploaded": c.get("uploaded", {}),
            "coded": {"A": len(A), "B": len(B), "C": len(c.get("C", {}))}, "sheet": {k: key.get(k) for k in ("created", "seed", "n_turns")},
            "total": len(rows)}


def report_csv() -> bytes:
    rep = report()["rows"]
    if not rep:
        return _csv_bytes(["안내"], [["아직 계산할 코딩 결과가 없어요."]])
    head = list(rep[0].keys())
    return _csv_bytes(head, [[("" if r[h] is None else r[h]) for h in head] for r in rep])


def status() -> dict:
    key = _key(); rows = key.get("rows", [])
    by = {k: sum(1 for x in rows if x["kind"] == k) for k in KIND_LABEL}
    rep = report()
    return {"sheet": {"created": key.get("created"), "seed": key.get("seed"),
                      "n_turns": key.get("n_turns"), "total": len(rows), "by_kind": by},
            "coded": rep["coded"], "uploaded": rep["uploaded"], "report": rep["rows"],
            "disagree": len(_disagree_ids()), "unresolved": sum(1 for x in final_rows() if x["left"])}


# ---------------------------------------------------------------------------
# 불일치 합의와 최종 코드 — 분석(analysis.py)에 쓰는 값
# ---------------------------------------------------------------------------
def _disagree_ids() -> list:
    rows = _key().get("rows", []); c = _codes(); A, B = c.get("A", {}), c.get("B", {})
    return [r["id"] for r in rows if r["id"] in A and r["id"] in B
            and any(A[r["id"]][e] != B[r["id"]][e] for e in ELEMENTS)]


def disagree_csv() -> bytes:
    """A·B가 다르게 본 발화만 모은 합의용 시트. 조건은 여전히 가린다. 빈칸에 합의한 값을 적어 코더 C로 올린다."""
    rows = {r["id"]: r for r in _key().get("rows", [])}
    c = _codes(); A, B = c.get("A", {}), c.get("B", {})
    out = []
    for i in _disagree_ids():
        r = rows[i]
        out.append([i, KIND_LABEL[r["kind"]], r["context"], r["text"],
                    *[f"{A[i][e]}/{B[i][e]}" for e in ELEMENTS], *[A[i][e] if A[i][e] == B[i][e] else "" for e in ELEMENTS], ""])
    return _csv_bytes(["코딩ID", "종류", "앞 대화", "판정할 발화", *[f"A/B_{e}" for e in ELEMENTS], *ELEMENTS, "메모"], out)


def final_rows() -> list:
    """요소마다 A=B면 그 값, 다르면 C(합의) 값, 합의가 없으면 빈칸."""
    c = _codes(); A, B, C = c.get("A", {}), c.get("B", {}), c.get("C", {})
    out = []
    for r in _key().get("rows", []):
        i = r["id"]
        if i not in A or i not in B:
            continue
        fin, left = {}, 0
        for e in ELEMENTS:
            if A[i][e] == B[i][e]:
                fin[e] = A[i][e]
            elif i in C:
                fin[e] = C[i][e]
            else:
                fin[e] = ""; left += 1
        out.append({"r": r, "final": fin, "left": left})
    return out


def final_csv() -> bytes:
    """관리자 보관용 — 최종 코드에 학생·조건을 붙인 표. analysis.py가 이 파일을 읽는다."""
    return _csv_bytes(["코딩ID", "종류", "학생", "학급", "차시", "턴", "전이회차", "전이문항",
                       "지원방식", "표시등", "점검요청", *[f"최종_{e}" for e in ELEMENTS], "합의남음",
                       *[f"자동_{e}" for e in ELEMENTS]],
                      [[x["r"]["id"], KIND_LABEL[x["r"]["kind"]], x["r"]["sid"], x["r"]["cond"].get("학급", ""),
                        x["r"]["session"], x["r"]["turn"], transfer.WAVE_LABEL.get(x["r"]["wave"], ""), x["r"]["item"],
                        x["r"]["cond"].get("지원방식", ""), x["r"]["cond"].get("표시등", ""),
                        x["r"]["cond"].get("점검요청", ""), *[x["final"][e] for e in ELEMENTS], x["left"],
                        *[x["r"]["auto"].get(e, "") for e in ELEMENTS]] for x in final_rows()])
