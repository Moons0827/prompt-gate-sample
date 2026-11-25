# -*- coding: utf-8 -*-
"""
학급 명단(고정 학생 계정)과 실험 조건(2×2) 관리
================================================

고정 학생 계정
  · 관리자가 학급을 만들고 학생 수를 넣으면 번호(01, 02, …)가 생긴다.
  · 학생은 이름을 쓰지 않고 '우리 반 → 내 번호(→ 입장 번호)'를 골라 들어온다.
  · 서버에 남는 학생 식별자는 가명 코드 '학급코드-번호'(예: A-07)뿐이다.
    번호와 실명의 대응표는 담임이 서버 밖에서 따로 보관한다.
  · 명단에 없는 번호로는 들어올 수 없으므로, 이름을 다르게 적어 같은 학생이
    두 사람으로 기록되던 문제(예비실험 5건)가 생기지 않는다.

실험 조건(설계 요소 두 가지를 켜고 끈다)
  · lights : 표시등(상황·조건·목적·대상 점등)을 학생 화면에 보여 주는가
  · check2 : 2차시에 AI 답 끝에 점검 요청 한마디를 붙이는가
  · support: AI 지원 방식 — "fade"(기본, 1차시만 되묻고 2~4차시는 답만) 또는
             "always"(1~4차시 내내 1차시처럼 직접 되묻는 비교 조건). always면 check2는 쓰지 않는다.
             지원을 줄여서 생긴 변화와 도구에 익숙해져서 생긴 변화를 가르기 위한 조건이다.
  배정 단위는 학급마다 고른다.
  · unit="class"   : 학급 전체가 한 조건(학급 단위 배정)
  · unit="student" : 학생마다 조건(같은 학급 안 무선 배정)
  조건은 매 턴 로그에 함께 남는다. 표시등이 꺼진 조건에서도 요소 판정은 뒤에서 그대로 돈다.

저장: store.get_config/put_config("roster") — 대화 초기화와 무관하게 유지된다.
"""
from __future__ import annotations
import os, re, time, random, secrets, threading

import store

KEY = "roster"
_LOCK = threading.RLock()
_CACHE: dict | None = None

CONDITIONS = [  # (lights, check2) — 2×2
    (True, True), (True, False), (False, True), (False, False),
]
COND_LABEL = {
    (True, True): "표시등 켬 · 점검 요청 켬",
    (True, False): "표시등 켬 · 점검 요청 끔",
    (False, True): "표시등 끔 · 점검 요청 켬",
    (False, False): "표시등 끔 · 점검 요청 끔",
}
SUPPORTS = ("fade", "always")
ALWAYS_ARM = (True, False, "always")   # 계속 되묻기 조건: 표시등은 켜고(지원 감소 조건과 같은 화면), 점검 요청은 해당 없음


def cond_label(lights: bool, check2: bool, support: str = "fade") -> str:
    if support == "always":
        return "계속 되묻기 · 표시등 " + ("켬" if lights else "끔")
    return COND_LABEL[(bool(lights), bool(check2))]


def _arms(include_always: bool) -> list:
    arms = [(lt, ck, "fade") for lt, ck in CONDITIONS]
    return arms + [ALWAYS_ARM] if include_always else arms

# 명단에 없는 학생(예: 학급을 만들지 않고 이름으로 들어온 시연·연수용 접속)의 조건.
DEFAULT_LIGHTS = os.environ.get("DEFAULT_LIGHTS", "1") == "1"
DEFAULT_CHECK2 = os.environ.get("DEFAULT_CHECK2", "1") == "1"
DEFAULT_SUPPORT = "always" if os.environ.get("DEFAULT_SUPPORT", "fade") == "always" else "fade"

_CID_RE = re.compile(r"^[A-Za-z0-9가-힣]{1,8}$")
_SID_RE = re.compile(r"^(?P<cid>[A-Za-z0-9가-힣]{1,8})-(?P<num>\d{2})$")


# ---------------------------------------------------------------------------
# 저장·불러오기
# ---------------------------------------------------------------------------
def _load() -> dict:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            try:
                d = store.get_config(KEY, strict=True) or {}
            except Exception:
                # DB가 잠깐 안 될 때 '명단 없음'으로 굳어 이름 입장이 열리지 않도록, 캐시하지 않고
                # 이번 요청만 빈 값으로 처리한다(다음 요청에서 다시 읽음).
                return {"classes": {}, "audit": [], "_unavailable": True}
            d.setdefault("classes", {})
            d.setdefault("audit", [])
            _CACHE = d
        return _CACHE


def _save(d: dict):
    with _LOCK:
        d["audit"] = d.get("audit", [])[-500:]   # 기록은 최근 500건만
        store.put_config(KEY, d)


def _audit(d: dict, action: str, **detail):
    d.setdefault("audit", []).append(
        {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action, **detail})


def reload():
    """저장소에서 다시 읽는다(다른 곳에서 바뀌었을 때)."""
    global _CACHE
    with _LOCK:
        _CACHE = None
    return _load()


def has_classes() -> bool:
    d = _load()
    # 명단을 읽지 못한 상태면 '명단 있음'으로 본다 → 이름 입장이 잘못 열리지 않게(학생은 잠시 뒤 다시 시도)
    return bool(d["classes"]) or bool(d.get("_unavailable"))


# ---------------------------------------------------------------------------
# 학급 만들기·지우기
# ---------------------------------------------------------------------------
def _pin() -> str:
    return f"{secrets.randbelow(10000):04d}"


def _cond(lights: bool | None, check2: bool | None, support: str | None = None) -> dict:
    return {"lights": lights, "check2": check2, "support": support}


def create_class(label: str, n: int, unit: str = "class", use_pin: bool = False) -> dict:
    label = (label or "").strip()
    if not _CID_RE.match(label):
        return {"error": "학급 코드는 글자·숫자 1~8자로 적어 주세요(띄어쓰기·기호 없이). 예: A, 5반1"}
    try:
        n = int(n)
    except Exception:
        return {"error": "학생 수를 숫자로 적어 주세요."}
    if not 1 <= n <= 60:
        return {"error": "학생 수는 1~60명이에요."}
    if unit not in ("class", "student"):
        unit = "class"
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        if label in d["classes"]:
            return {"error": f"'{label}' 학급이 이미 있어요."}
        d["classes"][label] = {
            "label": label, "n": n, "unit": unit, "pin": bool(use_pin),
            "cond": _cond(None, None),              # 학급 단위 조건(미배정이면 None)
            "students": {f"{i:02d}": {"pin": _pin(), "cond": _cond(None, None)}
                         for i in range(1, n + 1)},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _audit(d, "create_class", cid=label, n=n, unit=unit, pin=bool(use_pin))
        _save(d)
    return {"ok": True, "cid": label}


def delete_class(cid: str) -> dict:
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        if cid not in d["classes"]:
            return {"error": "없는 학급이에요."}
        del d["classes"][cid]
        _audit(d, "delete_class", cid=cid)
        _save(d)
    return {"ok": True}


def resize_class(cid: str, n: int) -> dict:
    """학생 수를 늘리거나 줄인다(줄일 때는 뒤 번호부터 — 이미 대화한 번호는 확인 후 줄일 것)."""
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        c = d["classes"].get(cid)
        if not c:
            return {"error": "없는 학급이에요."}
        n = int(n)
        if not 1 <= n <= 60:
            return {"error": "학생 수는 1~60명이에요."}
        st = c["students"]
        for i in range(1, n + 1):
            st.setdefault(f"{i:02d}", {"pin": _pin(), "cond": _cond(None, None)})
        for k in [k for k in st if int(k) > n]:
            del st[k]
        c["n"] = n
        _audit(d, "resize_class", cid=cid, n=n)
        _save(d)
    return {"ok": True}


# ---------------------------------------------------------------------------
# 조건 설정
# ---------------------------------------------------------------------------
def set_class(cid: str, unit: str | None = None, lights=None, check2=None,
              use_pin=None, support=None) -> dict:
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        c = d["classes"].get(cid)
        if not c:
            return {"error": "없는 학급이에요."}
        changed = {}
        if unit in ("class", "student") and unit != c["unit"]:
            c["unit"] = unit; changed["unit"] = unit
        if lights is not None:
            c["cond"]["lights"] = bool(lights); changed["lights"] = bool(lights)
        if check2 is not None:
            c["cond"]["check2"] = bool(check2); changed["check2"] = bool(check2)
        if use_pin is not None:
            c["pin"] = bool(use_pin); changed["pin"] = bool(use_pin)
        if support in SUPPORTS:
            c["cond"]["support"] = support; changed["support"] = support
        if changed:
            _audit(d, "set_class", cid=cid, **changed)
            _save(d)
    return {"ok": True}


def set_student(cid: str, num: str, lights=None, check2=None, support=None) -> dict:
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        c = d["classes"].get(cid)
        if not c:
            return {"error": "없는 학급이에요."}
        num = f"{int(num):02d}"
        s = c["students"].get(num)
        if not s:
            return {"error": "없는 번호예요."}
        if lights is not None:
            s["cond"]["lights"] = bool(lights)
        if check2 is not None:
            s["cond"]["check2"] = bool(check2)
        if support in SUPPORTS:
            s["cond"]["support"] = support
        _audit(d, "set_student", sid=f"{cid}-{num}", lights=s["cond"]["lights"],
               check2=s["cond"]["check2"], support=s["cond"].get("support"))
        _save(d)
    return {"ok": True}


def randomize_students(cid: str, seed: int | None = None, include_always: bool = False) -> dict:
    """같은 학급 안에서 학생을 네 조건(include_always면 계속 되묻기를 더한 다섯 조건)에 무선 배정.
       블록 무선화로 조건별 인원 차가 최대 1이 되게 하고, seed를 기록해 같은 배정을 다시 만들 수 있게 한다."""
    if seed is None:
        seed = secrets.randbelow(10**9)
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        c = d["classes"].get(cid)
        if not c:
            return {"error": "없는 학급이에요."}
        nums = sorted(c["students"], key=int)
        rng = random.Random(int(seed))
        seq = []
        while len(seq) < len(nums):          # 조건 수만큼 블록을 섞어 이어 붙임 → 조건별 인원 차 최대 1
            block = _arms(include_always)
            rng.shuffle(block)
            seq.extend(block)
        for num, (lt, ck, sup) in zip(nums, seq):
            c["students"][num]["cond"] = _cond(lt, ck, sup)
        c["unit"] = "student"
        c["student_seed"] = int(seed)
        c["student_arms"] = 5 if include_always else 4
        _audit(d, "randomize_students", cid=cid, seed=int(seed), n=len(nums),
               arms=5 if include_always else 4)
        _save(d)
    return {"ok": True, "seed": int(seed)}


def randomize_classes(cids: list[str] | None = None, seed: int | None = None,
                      include_always: bool = False) -> dict:
    """학급 단위 배정 학급들에 네 조건(include_always면 다섯 조건)을 고르게 무선 배정(차이 최대 1)."""
    if seed is None:
        seed = secrets.randbelow(10**9)
    with _LOCK:
        d = _load()
        if d.get("_unavailable"):
            return {"error": "지금 저장소에 연결되지 않아요. 잠시 뒤 다시 해 주세요."}
        pool = [cid for cid, c in d["classes"].items()
                if c["unit"] == "class" and (not cids or cid in cids)]
        if not pool:
            return {"error": "학급 단위로 배정할 학급이 없어요."}
        pool.sort()
        rng = random.Random(int(seed))
        rng.shuffle(pool)
        seq = []
        while len(seq) < len(pool):
            block = _arms(include_always)
            rng.shuffle(block)
            seq.extend(block)
        result = {}
        for cid, (lt, ck, sup) in zip(pool, seq):
            d["classes"][cid]["cond"] = _cond(lt, ck, sup)
            d["classes"][cid]["class_seed"] = int(seed)
            result[cid] = cond_label(lt, ck, sup)
        _audit(d, "randomize_classes", seed=int(seed), result=result,
               arms=5 if include_always else 4)
        _save(d)
    return {"ok": True, "seed": int(seed), "result": result}


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------
def parse_sid(sid: str):
    m = _SID_RE.match(sid or "")
    if not m:
        return None, None
    return m.group("cid"), m.group("num")


def is_roster_sid(sid: str) -> bool:
    cid, num = parse_sid(sid)
    if not cid:
        return False
    c = _load()["classes"].get(cid)
    return bool(c and num in c["students"])


def condition_for(sid: str) -> dict:
    """학생 한 명의 현재 조건. 매 턴 로그에 그대로 남긴다.
       assigned=False면 아직 조건이 정해지지 않은 상태(기본값으로 돌지만 로그에 표시)."""
    cid, num = parse_sid(sid)
    c = _load()["classes"].get(cid) if cid else None
    if not c or num not in c["students"]:
        return {"lights": DEFAULT_LIGHTS, "check2": DEFAULT_CHECK2, "support": DEFAULT_SUPPORT,
                "unit": "none", "cid": "", "assigned": False, "label": "명단 밖(기본값)"}
    src = c["cond"] if c["unit"] == "class" else c["students"][num]["cond"]
    lt, ck, sup = src.get("lights"), src.get("check2"), src.get("support")
    assigned = lt is not None and ck is not None
    lt = DEFAULT_LIGHTS if lt is None else bool(lt)
    ck = DEFAULT_CHECK2 if ck is None else bool(ck)
    sup = sup if sup in SUPPORTS else DEFAULT_SUPPORT
    if sup == "always":
        ck = False                       # 계속 되묻기 조건에는 2차시 점검 요청이 없다
    return {"lights": lt, "check2": ck, "support": sup, "unit": c["unit"], "cid": cid,
            "assigned": assigned,
            "label": cond_label(lt, ck, sup) + ("" if assigned else " (미배정·기본값)")}


def resolve_login(cid: str, num, pin: str = "") -> dict:
    """학생 입장 — 학급·번호(·입장 번호)를 확인하고 가명 코드를 돌려준다."""
    c = _load()["classes"].get((cid or "").strip())
    if not c:
        return {"ok": False, "error": "반을 다시 골라줘."}
    try:
        num = f"{int(num):02d}"
    except Exception:
        return {"ok": False, "error": "번호를 다시 골라줘."}
    s = c["students"].get(num)
    if not s:
        return {"ok": False, "error": "번호를 다시 골라줘."}
    if c.get("pin") and (pin or "").strip() != s["pin"]:
        return {"ok": False, "error": "입장 번호가 맞지 않아. 선생님께 받은 카드를 다시 봐줘."}
    return {"ok": True, "sid": f"{c['label']}-{num}"}


def public_classes() -> list:
    """학생 입장 화면용 — 학급 코드·인원·입장 번호 사용 여부만(조건·입장 번호 값은 숨김)."""
    out = []
    for cid, c in sorted(_load()["classes"].items()):
        out.append({"cid": cid, "n": c["n"], "pin": bool(c.get("pin")),
                    "nums": sorted(c["students"], key=int)})
    return out


def admin_view() -> dict:
    """관리자 화면용 — 학급, 학생별 입장 번호와 조건, 배정 기록."""
    d = _load()
    classes = []
    for cid, c in sorted(d["classes"].items()):
        studs = []
        for num in sorted(c["students"], key=int):
            cond = condition_for(f"{cid}-{num}")
            studs.append({"num": num, "sid": f"{cid}-{num}", "pin": c["students"][num]["pin"],
                          "lights": cond["lights"], "check2": cond["check2"],
                          "support": cond["support"], "assigned": cond["assigned"]})
        counts = {}
        for s in studs:
            k = cond_label(s["lights"], s["check2"], s["support"]) if s["assigned"] else "미배정"
            counts[k] = counts.get(k, 0) + 1
        classes.append({"cid": cid, "n": c["n"], "unit": c["unit"], "pin": bool(c.get("pin")),
                        "cond": c["cond"], "students": studs, "counts": counts,
                        "student_seed": c.get("student_seed"),
                        "class_seed": c.get("class_seed"), "created": c.get("created")})
    return {"classes": classes, "audit": d.get("audit", [])[-50:]}


def export_rows() -> list:
    """내보내기용 — 학생별 배정 결과(입장 번호는 빼고)."""
    rows = []
    for c in admin_view()["classes"]:
        for s in c["students"]:
            rows.append({"학생": s["sid"], "학급": c["cid"], "번호": s["num"],
                         "배정단위": "학급" if c["unit"] == "class" else "학생",
                         "표시등": int(s["lights"]), "점검요청": int(s["check2"]),
                         "지원방식": "계속되묻기" if s["support"] == "always" else "단계적감소",
                         "배정됨": int(s["assigned"]),
                         "씨앗": (c["class_seed"] if c["unit"] == "class" else c["student_seed"]) or ""})
    return rows
