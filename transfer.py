# -*- coding: utf-8 -*-
"""
지원 없는 전이 과제(사전·사후·지연)
===================================

AI의 답도, 표시등도 없이 학생이 'AI에게 보낼 첫 질문'만 써서 내는 과제다.
프로그램 안에서 관찰한 변화(첫 발화에 맥락을 먼저 담는 것)가 지원이 없는 상황에서도
나타나는지, 시간이 지나도 남는지를 확인하는 데 쓴다.

  · 회차(wave): pre(사전) · post(사후) · delay(지연, 예: 2주 뒤)
  · 회차마다 상황 카드 두 개. 카드에는 학생이 '알고 있는 사실'이 적혀 있고,
    학생은 그중 무엇을 질문에 담을지 스스로 고른다(요소 이름은 어디에도 쓰지 않는다).
  · 세 회차의 카드는 상황만 다르고, 담을 수 있는 정보의 종류와 양(상황·조건·목적·대상 각 1개)이 같게 만들었다.
  · 관리자가 회차를 열어야 학생이 들어올 수 있다. 한 번에 한 회차만 여는 것을 권한다.
  · 학생 화면에는 조건(표시등·점검 요청·지원 방식)과 관계없이 같은 화면이 나온다.
  · 제출한 글은 자동 코더로도 판정해 두지만(사람 코딩과 비교용), 학생에게는 보여 주지 않는다.

저장: store.get_config/put_config — "transfer"(설정·문항), "transfer_resp"(응답).
대화 초기화(reset)와 무관하게 남는다.
"""
from __future__ import annotations
import time, threading, copy

import store
import roster

CFG_KEY = "transfer"
RESP_KEY = "transfer_resp"
WAVES = ("pre", "post", "delay")
WAVE_LABEL = {"pre": "사전", "post": "사후", "delay": "지연"}
_LOCK = threading.RLock()

INSTRUCTION = ("아래 상황에서 AI에게 도움을 받으려고 해. AI에게 보낼 첫 질문을 한 번에 써 봐. "
               "이번에는 AI의 답이 나오지 않아. 다 쓰면 '제출하기'를 눌러 줘.")

# 카드의 네 줄은 각각 상황·조건·목적·대상에 해당하지만, 학생 화면에는 요소 이름 없이 사실만 보인다.
# key는 채점용 기준(관리자·코더용)이다.
DEFAULT_ITEMS = {
    "pre": [
        {"id": "pre1", "title": "동생 생일 선물",
         "facts": ["다음 주 토요일이 동생 생일이야.",
                   "용돈이 1만 원밖에 없어.",
                   "동생을 깜짝 놀라게 해 주고 싶어.",
                   "동생은 여덟 살이고 공룡을 아주 좋아해."],
         "key": {"상황": "다음 주 토요일 동생 생일", "조건": "1만 원 이내",
                 "목적": "깜짝 놀라게 하기", "대상": "여덟 살, 공룡을 좋아하는 동생"}},
        {"id": "pre2", "title": "모둠 발표 준비",
         "facts": ["다음 주 사회 시간에 모둠 발표가 있어.",
                   "발표는 3분 안에 끝내야 하고 그림 자료는 한 장만 쓸 수 있어.",
                   "우리 동네 문화재를 친구들에게 알리고 싶어.",
                   "듣는 사람은 같은 반 친구들이야."],
         "key": {"상황": "다음 주 사회 시간 모둠 발표", "조건": "3분, 그림 한 장",
                 "목적": "동네 문화재 알리기", "대상": "같은 반 친구들"}},
    ],
    "post": [
        {"id": "post1", "title": "할머니 생신 카드",
         "facts": ["이번 주말에 할머니 생신 가족 모임이 있어.",
                   "카드에는 다섯 줄까지만 쓸 수 있어.",
                   "할머니께 고마운 마음을 전하고 싶어.",
                   "할머니는 눈이 조금 안 좋으시고 손주 이야기를 좋아하셔."],
         "key": {"상황": "이번 주말 할머니 생신 모임", "조건": "다섯 줄 이내",
                 "목적": "고마운 마음 전하기", "대상": "눈이 안 좋고 손주 이야기를 좋아하는 할머니"}},
        {"id": "post2", "title": "체육대회 응원 구호",
         "facts": ["다음 달에 학급 체육대회가 있어.",
                   "구호는 10초 안에 다 같이 외칠 수 있어야 해.",
                   "우리 반이 하나로 뭉치면 좋겠어.",
                   "우리 반 친구들은 노래하는 걸 좋아해."],
         "key": {"상황": "다음 달 학급 체육대회", "조건": "10초 안에 외치기",
                 "목적": "반이 하나로 뭉치기", "대상": "노래를 좋아하는 우리 반 친구들"}},
    ],
    "delay": [
        {"id": "delay1", "title": "친구와 화해",
         "facts": ["어제 쉬는 시간에 친구랑 말다툼을 했어.",
                   "직접 말하기는 어려워서 문자로 보내려고 해.",
                   "다시 친하게 지내고 싶어.",
                   "그 친구는 장난치는 걸 좋아하고 한번 삐지면 오래 가."],
         "key": {"상황": "어제 쉬는 시간 말다툼", "조건": "문자로 보내기",
                 "목적": "다시 친하게 지내기", "대상": "장난을 좋아하고 오래 삐지는 친구"}},
        {"id": "delay2", "title": "주말 가족 나들이",
         "facts": ["이번 주말에 가족과 나들이를 가기로 했어.",
                   "집에서 한 시간 안에 갈 수 있어야 하고, 비 소식이 있어.",
                   "가족 모두 즐거운 하루를 보내고 싶어.",
                   "네 살 동생도 함께 가."],
         "key": {"상황": "이번 주말 가족 나들이", "조건": "한 시간 이내, 비 소식",
                 "목적": "모두 즐거운 하루", "대상": "네 살 동생과 가족"}},
    ],
}


# ---------------------------------------------------------------------------
def _cfg() -> dict:
    d = store.get_config(CFG_KEY) or {}
    d.setdefault("open", {w: False for w in WAVES})
    d.setdefault("items", copy.deepcopy(DEFAULT_ITEMS))
    for w in WAVES:
        d["open"].setdefault(w, False)
        d["items"].setdefault(w, copy.deepcopy(DEFAULT_ITEMS[w]))
    return d


def _resp() -> dict:
    return store.get_config(RESP_KEY) or {}


def open_wave() -> str | None:
    """지금 열린 회차(여러 개면 사전→사후→지연 순으로 앞의 것)."""
    o = _cfg()["open"]
    return next((w for w in WAVES if o.get(w)), None)


def set_open(wave: str, on: bool) -> dict:
    if wave not in WAVES:
        return {"error": "회차가 올바르지 않아요."}
    with _LOCK:
        d = _cfg()
        d["open"][wave] = bool(on)
        d.setdefault("audit", []).append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "action": "open" if on else "close", "wave": wave})
        d["audit"] = d["audit"][-100:]
        store.put_config(CFG_KEY, d)
    return {"ok": True, "open": d["open"]}


def set_items(wave: str, items: list) -> dict:
    """관리자가 카드 문구를 고친다. 응답이 이미 있는 회차는 고치지 못하게 막는다(문항이 바뀌면 비교가 깨짐)."""
    if wave not in WAVES:
        return {"error": "회차가 올바르지 않아요."}
    if not isinstance(items, list) or not items:
        return {"error": "문항 형식이 올바르지 않아요."}
    clean = []
    for i, it in enumerate(items, 1):
        facts = [str(f).strip() for f in (it.get("facts") or []) if str(f).strip()]
        if not facts:
            return {"error": f"{i}번 카드에 사실이 비어 있어요."}
        clean.append({"id": str(it.get("id") or f"{wave}{i}")[:20], "title": str(it.get("title") or "")[:40],
                      "facts": facts[:8], "key": it.get("key") or {}})
    with _LOCK:
        if any(wave in v for v in _resp().values()):
            return {"error": "이미 응답이 있는 회차라 문항을 바꿀 수 없어요."}
        d = _cfg()
        d["items"][wave] = clean
        store.put_config(CFG_KEY, d)
    return {"ok": True}


def student_view(sid: str) -> dict:
    """학생 화면용 — 열린 회차의 카드(채점 기준 key는 빼고)와 이미 낸 문항."""
    w = open_wave()
    if not w:
        return {"ok": True, "wave": None}
    items = _cfg()["items"][w]
    done = set((_resp().get(sid) or {}).get(w, {}).keys())
    return {"ok": True, "wave": w, "wave_label": WAVE_LABEL[w], "instruction": INSTRUCTION,
            "items": [{"id": it["id"], "title": it["title"], "facts": it["facts"],
                       "done": it["id"] in done} for it in items]}


def submit(sid: str, wave: str, item_id: str, text: str, meta: dict | None = None,
           auto: dict | None = None) -> dict:
    text = (text or "").strip()
    if not roster.is_roster_sid(sid):
        return {"error": "반과 번호로 다시 들어와 줘."}
    if wave != open_wave():
        return {"error": "지금은 이 과제를 낼 수 없어. 선생님께 여쭤 봐."}
    if not text:
        return {"error": "질문을 써 줘."}
    items = {it["id"]: it for it in _cfg()["items"][wave]}
    if item_id not in items:
        return {"error": "문항을 다시 골라 줘."}
    with _LOCK:
        r = _resp()
        slot = r.setdefault(sid, {}).setdefault(wave, {})
        if item_id in slot:
            return {"error": "이 문항은 이미 냈어."}
        m = meta or {}
        slot[item_id] = {"text": text[:1000], "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "compose_ms": m.get("compose_ms"), "revisions": m.get("revisions"),
                         "deleted_chars": m.get("deleted_chars"), "pastes": m.get("pastes"),
                         "auto": auto or {}}
        store.put_config(RESP_KEY, r)
    return {"ok": True}


def admin_view() -> dict:
    d = _cfg(); r = _resp()
    counts = {w: {it["id"]: 0 for it in d["items"][w]} for w in WAVES}
    for sid, waves in r.items():
        for w, its in waves.items():
            for iid in its:
                if w in counts:
                    counts[w][iid] = counts[w].get(iid, 0) + 1
    return {"open": d["open"], "items": d["items"], "counts": counts,
            "labels": WAVE_LABEL, "audit": d.get("audit", [])[-20:]}


def all_responses() -> list:
    """내보내기·코딩용 — 응답 한 줄씩."""
    d = _cfg(); r = _resp()
    titles = {it["id"]: it.get("title", "") for w in WAVES for it in d["items"][w]}
    out = []
    for sid in sorted(r):
        for w in WAVES:
            for iid, v in sorted((r[sid].get(w) or {}).items()):
                out.append({"sid": sid, "wave": w, "item": iid, "title": titles.get(iid, ""), **v})
    return out
