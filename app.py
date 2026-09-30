# -*- coding: utf-8 -*-
"""
prompt-gate 확장판 — 샘플 서버 (표준 라이브러리만 사용)
  · 학생 화면(static/index.html) 서빙
  · POST /api/turn : 한 턴 처리(안전→탐지→넛지→AI응답→로그 적재)
  · 턴 로그는 logs/turns.csv 에 코딩 시트 스키마로 누적

실행:  python app.py   →  http://localhost:8000
실 LLM:  export ANTHROPIC_API_KEY=... (없으면 mock 모드로 동작)

prompt-gate(FastAPI)로 옮길 때: 아래 handle_turn 로직을 그대로 라우트에 넣으면 됨.
detector.py / llm.py 는 프레임워크 독립이라 수정 없이 재사용.
"""
import os, json, csv, time, collections, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from detector import ConversationCoder, NudgePolicy, LOG_FIELDS, SafetyFilter, TurnLog, ELEMENTS
import llm
import store

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
LOGDIR = os.path.join(HERE, "logs")
os.makedirs(LOGDIR, exist_ok=True)
LOGCSV = os.path.join(LOGDIR, "turns.csv")
TRANSCRIPT = os.path.join(LOGDIR, "transcripts.jsonl")

# 연속 전송 방지(초). 실수 연타·도배·비용 폭주를 막는다. 기본 5초.
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", 5))

# /s (차시 번호 없이 접속) 시 기본 차시.
LESSON_SESSION = int(os.environ.get("LESSON_SESSION", 1))

# 안전 계층: Layer1 키워드(입력)·Layer2 키워드(AI 출력)는 항상 켜짐.
# Layer3 = LLM 2차 입력 점검(우회·간접 표현 보완). 턴당 LLM 호출이 1회 늘어 비용↑ →
# 기본 꺼짐. 실제 학생 투입 시 SAFETY_LLM=1 권장.
SAFETY_LLM = os.environ.get("SAFETY_LLM", "0") == "1"

# 접근 코드 — 실제 운영 시 반드시 환경변수로 바꾸세요.
#  · 관리자(나): 메인 허브(1~4차시 전체 + 모든 대시보드) 접근
#  · 교사: 자기 차시 대시보드만 접근
ADMIN_KEY   = os.environ.get("ADMIN_KEY",   "corelab-admin")
TEACHER_KEY = os.environ.get("TEACHER_KEY", "corelab-teacher")


def _load_dotenv():
    """같은 폴더의 .env 파일을 읽어 환경변수로 넣는다(이미 설정된 값은 유지)."""
    p = os.path.join(HERE, ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

# (student_id, session) -> {coder, messages, ...}  (실행 중 캐시; 영속본은 store)
# OrderedDict + 상한(LRU) → 메모리 초과(OOM)로 인스턴스가 죽는 것을 막는다.
# Neon(영속본)이 진짜 원본이므로, 캐시에서 밀려난 세션도 다음 접속 때 그대로 복원된다.
SESSIONS = collections.OrderedDict()
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 80))   # 캐시에 동시에 둘 최대 세션 수
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", 30))   # LLM에 보내는 최근 대화 길이 상한
                                                          # (transcript에는 전체가 남으므로 기록 손실 없음)

# 동일 이름 동시 접속 차단 — 메모리 전용(영속화 안 함, 작음, LRU 대상 아님).
# (student_id, session) -> {"token": str, "seen": float}
PRESENCE = {}
JOIN_WINDOW = int(os.environ.get("JOIN_WINDOW", 20))     # 이 시간(초) 안에 활동이 있으면 '접속 중'으로 봄

# 비용/폭주 방지: 한 학생이 한 차시에서 보낼 수 있는 최대 턴 수(초과 시 부드럽게 멈춤).
MAX_STUDENT_TURNS = int(os.environ.get("MAX_STUDENT_TURNS", 60))
# 관리자 비상 정지 — True면 모든 학생의 새 턴 처리를 멈춘다(/api/pause로 토글).
PAUSED = False
# 자해·정서위험 범주 — 이 경우엔 학생에게 조용한 대기 대신 따뜻한 '케어' 안내를 보낸다.
CARE_CATEGORY = "자해·정서위험"


def _cache_put(key, S):
    """세션을 캐시에 넣고 최신으로 표시. 상한을 넘으면 가장 오래된 것부터 내보낸다(영속본은 유지)."""
    SESSIONS[key] = S
    SESSIONS.move_to_end(key)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)     # 가장 오래 안 쓴 세션 제거(다음 접속 때 store에서 복원)


def _cap_messages(S):
    """LLM 호출 비용/메모리를 위해 messages를 최근 MAX_MESSAGES개로 제한(대화 맥락은 충분히 유지)."""
    m = S.get("messages")
    if m and len(m) > MAX_MESSAGES:
        S["messages"] = m[-MAX_MESSAGES:]


def _touch_presence(sid, session):
    """해당 이름이 '지금 접속 중'임을 갱신(턴 처리·폴링에서 호출 → 폴링이 심장박동 역할)."""
    if not sid:
        return
    key = (sid, int(session))
    p = PRESENCE.get(key)
    if p:
        p["seen"] = time.time()


def _is_kicked(sid, session, token) -> bool:
    """이 토큰이 더 이상 이 이름의 '소유자'가 아니면 True(다른 기기가 이어받음)."""
    p = PRESENCE.get((sid, int(session)))
    return bool(p and token and token != p.get("token"))


def handle_join(body):
    """학생 입장 — 동일 이름은 '이어받기(takeover)':
       같은 이름으로 이미 접속 중이어도 막지 않고, 새 접속이 소유권을 가져온다.
       기존 접속(이전 탭/기기)은 폴링에서 kicked 신호를 받아 종료된다.
       과거 대화(transcript)·요소·직접대화를 함께 돌려줘 화면에 그대로 복원한다."""
    import secrets
    sid = (body.get("student") or "").strip()
    session = int(body.get("session", 1))
    token = (body.get("token") or "").strip()
    if not sid:
        return {"ok": False, "error": "이름을 입력해줘."}
    key = (sid, session)
    now = time.time()
    p = PRESENCE.get(key)
    active = bool(p and (now - p.get("seen", 0) < JOIN_WINDOW))
    took_over = bool(active and token != p.get("token"))
    # 본인 새로고침(토큰 일치)이면 기존 토큰 유지, 그 외에는 새 토큰 발급 → 이 토큰만 유효
    if not token or took_over:
        token = secrets.token_hex(8)
    PRESENCE[key] = {"token": token, "seen": now}   # 이 순간 이전 토큰은 무효 → 이전 탭은 kicked
    # 과거 대화 복원
    S = _lookup_session(sid, session)
    hist, elems, direct = [], {e: False for e in ELEMENTS}, []
    if S:
        for t in S["transcript"]:
            hist.append({"student": t.get("student"), "ai": t.get("ai"),
                         "pending": bool(t.get("pending"))})
        elems = S["coder"].elements()
        direct = list(S.get("direct", []))
    return {"ok": True, "token": token, "took_over": took_over,
            "history": hist, "elements": elems, "direct": direct}


def _new_session(sid, session, topic):
    return {
        "coder": ConversationCoder(sid, int(session), topic or "자유",
                                   policy=NudgePolicy(session=int(session)),
                                   safety_llm=SAFETY_LLM),
        "messages": [], "updated": time.time(), "ack": True, "last": "",
        "last_elicited": False, "transcript": [], "last_turn_ts": 0,
        "outbox": [],          # 학생 폴링(/api/pending)으로 내보낼 메시지 큐(교사답변·허용·전체공지)
        "direct": [],          # 학생↔교사 1:1 직접 대화 [{from:'student'|'teacher', text, ts}]
        "call": False,         # 학생이 '선생님 부르기'를 눌렀는가(대시보드에 호출 표시)
        "call_ts": 0,
    }


def _serialize(S) -> dict:
    """세션 상태를 저장 가능한 dict로. 대화 복원에 필요한 코더 내부까지 포함."""
    from dataclasses import asdict
    c = S["coder"]
    return {
        "coder": {"history": c._history, "cum": c._cum, "turn": c._turn,
                  "logs": [asdict(l) for l in c.logs]},
        "messages": S["messages"], "ack": S["ack"], "last": S["last"],
        "last_elicited": S["last_elicited"], "updated": S["updated"],
        "last_turn_ts": S.get("last_turn_ts", 0), "transcript": S["transcript"],
        "outbox": S.get("outbox", []), "direct": S.get("direct", []),
        "call": S.get("call", False), "call_ts": S.get("call_ts", 0),
    }


def _deserialize(state: dict, sid, session, topic) -> dict:
    """store에서 읽은 dict를 실행용 세션(코더 포함)으로 복원."""
    S = _new_session(sid, session, topic)
    c = S["coder"]
    cd = state.get("coder", {})
    c._history = list(cd.get("history", []))
    cum = cd.get("cum", {})
    c._cum = {e: bool(cum.get(e, False)) for e in ELEMENTS}
    c._turn = int(cd.get("turn", 0))
    c.logs = [TurnLog(**d) for d in cd.get("logs", [])]
    S["messages"] = list(state.get("messages", []))
    S["ack"] = bool(state.get("ack", True))
    S["last"] = state.get("last", "")
    S["last_elicited"] = bool(state.get("last_elicited", False))
    S["updated"] = float(state.get("updated", time.time()))
    S["last_turn_ts"] = float(state.get("last_turn_ts", 0))
    S["transcript"] = list(state.get("transcript", []))
    # outbox: 구(舊)데이터의 pending_out 단일 슬롯도 흡수
    ob = list(state.get("outbox", []))
    if state.get("pending_out"):
        ob.append(state["pending_out"])
    S["outbox"] = ob
    S["direct"] = list(state.get("direct", []))
    S["call"] = bool(state.get("call", False))
    S["call_ts"] = float(state.get("call_ts", 0))
    return S


def _get_session(sid, session, topic):
    """캐시에 있으면 그대로, 없으면 store에서 복원(=재시작 후 이어가기), 그래도 없으면 신규."""
    key = (sid, int(session))
    if key not in SESSIONS:
        saved = store.get_state(sid, int(session))
        S = _deserialize(saved, sid, session, topic) if saved \
            else _new_session(sid, session, topic)
        _cache_put(key, S)
    else:
        SESSIONS.move_to_end(key)   # 최근 사용 표시(LRU)
    return SESSIONS[key]


def _dash_row(sid, session, S) -> dict:
    """대시보드 한 줄을 미리 계산(저장 시 함께 넣어 재시작 후에도 바로 조회)."""
    coder = S["coder"]; s = coder.summary()
    level, reason = compute_status(S)
    return {
        "student_id": sid, "session": int(session), "topic": coder.meta[2],
        "turns": s["총턴수"], "요소누적도": s["요소누적도"],
        "대화유형": s["대화유형"], "자발비율": s["자발비율"],
        "last": (S.get("last") or "")[:30], "level": level, "reason": reason,
        "updated": S["updated"], "call": bool(S.get("call")),
    }


def _save_session(sid, session, S):
    """세션 상태 + 대시보드 요약을 store에 영속화(매 턴 호출)."""
    store.put_state(sid, int(session), S["coder"].meta[2],
                    _serialize(S), _dash_row(sid, session, S))


def _append_transcript(entry: dict):
    try:
        with open(TRANSCRIPT, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# 물음표 없이 끝나도 되묻기로 볼 의문형 어미(문장 '끝'에서만 인정).
_Q_ENDINGS = ("니", "냐", "나요", "까", "까요", "래", "래요", "을래", "ㄹ래", "줄래",
              "어때", "어때요", "은지", "는지", "은가", "는가", "ㄴ가")
# 되묻기의 또 다른 형태 — '더 말해줘/알려줘'식으로 학생에게 정보를 더 청하는 유도.
_INVITE_ENDINGS = ("말해줘", "말해 줘", "알려줘", "알려 줘", "적어줘", "적어봐",
                   "말해줄래", "얘기해줘", "이야기해줘")
# 따옴표로 감싼 '예시'(예: "'이거 맞아?'처럼")의 물음표에 속지 않도록 제거하는 패턴.
_QUOTED = re.compile(r'[\'"“”‘’][^\'"“”‘’]*[\'"“”‘’]')
_TAIL_TRIM = re.compile(r'[\s"\'”’」』)\]】.!…~]+$')


def _asks(text: str) -> bool:
    """AI 응답이 학생에게 '되묻기/유도'로 기능했는지 판정.
       자발/유도 계기 판정 정확도를 위해 개선한 버전:
        · 따옴표 예시 속 물음표는 무시(오탐 방지),
        · 남은 본문에 진짜 물음표가 있으면 되묻기,
        · 물음표가 없어도 끝이 의문형 어미거나 '더 말해줘'식 유도면 되묻기."""
    if not text:
        return False
    t = text.strip()
    cleaned = _QUOTED.sub(" ", t)      # 예시 인용 제거
    if "?" in cleaned:                 # 예시가 아닌 실제 물음표 → 되묻기
        return True
    tail = _TAIL_TRIM.sub("", t)       # 끝 장식문자 정리 후 어미 검사
    return tail.endswith(_Q_ENDINGS) or tail.endswith(_INVITE_ENDINGS)


def compute_status(S: dict) -> tuple:
    """학생 상태를 3단계로 분류: 즉시(안전) / 주의(학습위험) / 로그(정상)."""
    coder = S["coder"]; logs = coder.logs
    if not S.get("ack") and any(l.개입 == "교사되돌림" for l in logs):
        cat = next((l.비고 for l in reversed(logs) if l.개입 == "교사되돌림"), "안전 신호")
        return "즉시", cat
    recent = logs[-3:]
    if sum(1 for l in recent if l.행위유형 == "단순반복") >= 2:
        return "주의", "반복 정체(후속 진전 없음)"
    if logs and coder.summary()["대화유형"] == "반복정체형":
        return "주의", "반복정체형"
    return "로그", "정상"


def dashboard_data(only_session=None) -> list:
    """store(영속본)에서 요약 행을 읽어 구성 → 재시작·다른 세션에도 그대로 보임."""
    rows = store.list_summaries(only_session)
    now = time.time()
    for r in rows:
        r["ago"] = int(now - r.get("updated", now))
    order = {"즉시": 0, "주의": 1, "로그": 2}
    rows.sort(key=lambda r: (order.get(r.get("level", "로그"), 2), r["ago"]))
    return rows


def _append_log(row: dict):
    from dataclasses import asdict
    new = not os.path.exists(LOGCSV)
    with open(LOGCSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def handle_turn(body: dict) -> dict:
    sid = body.get("student_id", "익명")
    session = body.get("session", 1)
    topic = body.get("topic", "자유")
    utterance = (body.get("utterance") or "").strip()
    if not utterance:
        return {"error": "빈 발화"}

    # 다른 기기가 같은 이름으로 이어받았으면 이 탭은 종료
    if _is_kicked(sid, session, (body.get("token") or "").strip()):
        return {"kicked": True}

    S = _get_session(sid, session, topic)
    _touch_presence(sid, session)   # 이 이름이 아직 접속 중임을 갱신

    # 관리자 비상 정지 — 새 턴을 처리하지 않고 안내만
    if PAUSED:
        return {"reply": "지금은 잠깐 쉬는 시간이야. 선생님 안내를 기다려줘!",
                "elements": S["coder"].elements(), "paused": True}
    # 학생당 턴 상한 — 비용 폭주·도배 방지(초과 시 LLM 호출 없이 부드럽게 멈춤)
    if S["coder"]._turn >= MAX_STUDENT_TURNS:
        return {"reply": "오늘 정말 많이 이야기했다! 잠깐 쉬었다가 선생님이 안내해줄 거야.",
                "elements": S["coder"].elements(), "capped": True}

    # 연속 전송 방지: 마지막 전송 후 COOLDOWN_SEC 이내면 처리하지 않음(서버측 강제)
    now = time.time()
    since = now - S.get("last_turn_ts", 0)
    if since < COOLDOWN_SEC:
        remain = int(COOLDOWN_SEC - since) + 1
        return {"error": f"조금만 천천히! {remain}초 뒤에 다시 보내요.", "cooldown": True}
    S["last_turn_ts"] = now

    coder, messages = S["coder"], S["messages"]

    # 직전 AI가 되물었는지를 이번 턴 계기 판정에 사용
    # (coder.step 안에서 Layer1 키워드 + Layer3 LLM 2차로 입력 안전을 점검)
    res = coder.step(utterance, elicited_prev=S["last_elicited"])
    from dataclasses import asdict
    L = res["log"]

    messages.append({"role": "user", "content": utterance})
    S["updated"] = time.time()
    S["last"] = utterance

    reply = None
    if res["safety"].ok:
        filled = sum(1 for v in coder.elements().values() if v)
        reply = llm.chat(messages, filled=filled, session=int(session))
        out_sf = SafetyFilter().check(reply or "")     # Layer 2: AI 출력 안전 점검
        if not out_sf.ok:
            reply = None                               # 위험 출력 차단 → 교사 호출
            res["safety"] = out_sf
            L.개입 = "교사되돌림"
            L.비고 = (L.비고 + f" 출력차단:{out_sf.category}").strip()
            S["ack"] = False
            S["last_elicited"] = False
        else:
            messages.append({"role": "assistant", "content": reply})
            S["last_elicited"] = _asks(reply)          # 이번 AI 응답이 되물었는가
    else:
        S["ack"] = False                               # 새 안전 알림 → 미확인
        S["last_elicited"] = False

    blocked = not res["safety"].ok    # 입력/출력 안전 차단 → 교사 개입 대기
    _append_log(asdict(L))      # 최종 로그(출력 차단 반영) 적재
    # 이번 턴에 '탐지된' 요소(누적 아님) — 사람 코딩과 대조해 코더 신뢰도(F1)를 재기 위해 별도 보관
    turn_el = {"상황": int(L.상황), "조건": int(L.조건), "목적": int(L.목적), "대상": int(L.대상)}
    entry = {"turn": L.턴, "student": utterance, "ai": reply,
             "계기": L.계기, "개입": L.개입, "행위유형": L.행위유형,
             "safety_ok": res["safety"].ok, "category": res["safety"].category,
             "elements": coder.elements(), "turn_elements": turn_el, "pending": blocked,
             "ts": time.strftime("%Y-%m-%d %H:%M:%S")}   # 턴별 시각(프로세스 분석용)
    S["transcript"].append(entry)
    _append_transcript({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "student_id": sid, "session": int(session), **entry})
    _cap_messages(S)                 # LLM 맥락/메모리 상한(전체 기록은 transcript에 보존)
    _save_session(sid, session, S)   # ★ 매 턴 즉시 영속화 (튕겨도 이어가기)

    if blocked:
        # 학생에겐 위험/차단 사실을 절대 노출하지 않는다 — '생각 중' 상태만 돌려주고,
        # 교사가 답을 넣거나 'AI 허용'을 누르면 학생 폴링(/api/pending)으로 자연스럽게 전달된다.
        # 단 자해·정서위험만은 조용한 대기 대신 '케어' 안내를 보이도록 신호(care)를 준다.
        # (구체 범주명은 노출하지 않는다 — 아이는 자기가 무엇으로 걸렸는지 알 수 없음)
        return {"status": "waiting", "elements": coder.elements(),
                "care": (res["safety"].category == CARE_CATEGORY)}
    return {"reply": reply, "elements": coder.elements(), "summary": coder.summary()}


# ---- 교사 개입: 답변 넣어주기 / 사소한 위험 통과(AI 허용) / 학생 폴링 -----------
def _lookup_session(sid, session):
    """폴링 등 조회 전용 — 캐시에 있으면 그대로, 없으면 store에서 복원. 없으면 None(빈 세션 생성 안 함)."""
    key = (sid, int(session))
    if key in SESSIONS:
        SESSIONS.move_to_end(key)
        return SESSIONS[key]
    st = store.get_state(sid, int(session))
    if st:
        S = _deserialize(st, sid, session, "자유")
        _cache_put(key, S)
        return S
    return None


def _deliver_reply(sid, session, reply_text, kind):
    """교사 답변/허용을 세션에 반영하고, 학생 폴링(/api/pending)으로 전달할 슬롯에 넣는다."""
    S = _get_session(sid, session, "자유")
    tr = S["transcript"]
    target = next((e for e in reversed(tr) if e.get("pending")), None)
    if target is not None:                      # 대기 중이던 차단 턴을 교사 답으로 채움
        target["ai"] = reply_text; target["pending"] = False; target["개입"] = kind
    else:                                        # 대기 턴이 없을 때의 일반 개입 — 새 항목
        tr.append({"turn": (tr[-1]["turn"] + 1 if tr else 1), "student": "(선생님 개입)",
                   "ai": reply_text, "계기": "없음", "개입": kind, "safety_ok": True,
                   "category": None, "elements": S["coder"].elements(), "pending": False})
    S["messages"].append({"role": "assistant", "content": reply_text})
    S.setdefault("outbox", []).append(reply_text)   # 학생 폴링이 순서대로 가져가 정상 AI 말풍선으로 표시
    S["ack"] = True
    S["last_elicited"] = _asks(reply_text)
    S["updated"] = time.time()
    _save_session(sid, session, S)


def handle_teacher_reply(sid, session, text):
    text = (text or "").strip()
    if not text:
        return {"error": "빈 답변"}
    _deliver_reply(sid, session, text, "교사답변")
    return {"ok": True}


def handle_allow_ai(sid, session):
    """사소한 위험 → 교사 판단으로 통과. AI가 정상적으로 답하게 만들어 학생에게 전달."""
    S = _get_session(sid, session, "자유")
    filled = sum(1 for v in S["coder"].elements().values() if v)
    reply = llm.chat(S["messages"], filled=filled, session=int(session))  # 교사 승인 → 안전 재검사 생략
    _deliver_reply(sid, session, reply, "교사허용")
    return {"ok": True}


def handle_pending(sid, session, token=""):
    """학생 폴링 — 큐(outbox)에 쌓인 교사답변·허용·전체공지를 순서대로 하나씩 전달.
       이어받기 종료(kicked)도 여기서 알린다."""
    if _is_kicked(sid, session, token):
        return {"kicked": True}
    _touch_presence(sid, session)   # 폴링이 곧 심장박동 — 접속 중 표시 갱신
    S = _lookup_session(sid, session)
    if not S:
        return {"reply": None}
    ob = S.get("outbox") or []
    if ob:
        reply = ob.pop(0)
        _save_session(sid, session, S)
        return {"reply": reply, "elements": S["coder"].elements()}
    return {"reply": None}


def handle_direct(sid, session, since=0):
    """1:1 직접 대화 조회 — since 이후의 메시지와 호출 상태를 돌려준다(학생·교사 공용)."""
    S = _lookup_session(sid, session)
    if not S:
        return {"direct": [], "total": 0, "call": False}
    d = S.get("direct") or []
    return {"direct": d[since:], "total": len(d), "call": bool(S.get("call"))}


def handle_call(sid, session):
    """학생이 '선생님 부르기'를 누름 → 호출 플래그 ON(대시보드에 표시)."""
    S = _get_session(sid, session, "자유")
    S["call"] = True; S["call_ts"] = time.time()
    _save_session(sid, session, S)
    return {"ok": True}


def handle_direct_send(sid, session, text, who):
    """1:1 직접 대화에 메시지 추가. who='student'|'teacher'. 교사가 답하면 호출 해제."""
    text = (text or "").strip()
    if not text:
        return {"error": "빈 메시지"}
    S = _get_session(sid, session, "자유")
    S.setdefault("direct", []).append({"from": who, "text": text,
                                       "ts": time.strftime("%H:%M")})
    if who == "student":
        S["call"] = True; S["call_ts"] = time.time()   # 학생이 말하면 호출로도 표시
    else:
        S["call"] = False                              # 교사가 응답하면 호출 해제
    S["updated"] = time.time()
    _save_session(sid, session, S)
    return {"ok": True, "direct_total": len(S["direct"])}


def handle_broadcast(text, only_session=None):
    """관리자/교사 전체 공지 — 모든(또는 해당 차시) 학생의 outbox에 안내 메시지를 넣는다."""
    text = (text or "").strip()
    if not text:
        return {"error": "빈 공지"}
    msg = "📢 [선생님 공지] " + text
    n = 0
    for r in store.list_all(only_session):     # 영속본 기준 전체 학생
        sid, sess = r["student_id"], r["session"]
        S = _get_session(sid, sess, r.get("topic") or "자유")
        S.setdefault("outbox", []).append(msg)
        S["updated"] = time.time()
        _save_session(sid, sess, S)
        n += 1
    return {"ok": True, "count": n}


# ---- 로그 내보내기 (관리자) ---------------------------------------------------
def _export_bytes(fmt, only_session, anon=True):
    """anon=True(기본): 학생 실명을 S01·S02…로 익명화(논문·공유 안전).
       anon=False: 관리자가 명시적으로 '실명 포함'을 선택했을 때만 실명 그대로."""
    import io, csv as _csv
    rows = store.list_all(only_session)   # [{student_id, session, topic, state}]
    sums = store.list_summaries(only_session)
    stamp = time.strftime("%Y%m%d")
    scope = "전체" if only_session is None else f"{only_session}차시"
    tag = "실명" if not anon else "익명"
    # 익명화 매핑: 스코프 내 모든 실명 → S01… (내보내기 종류가 달라도 같은 번호 → 파일 간 연결 가능)
    ids = sorted({r["student_id"] for r in rows} | {s.get("student_id") for s in sums})
    alias = {sid: f"S{i:02d}" for i, sid in enumerate(ids, 1)}
    nm = (lambda sid: sid) if not anon else (lambda sid: alias.get(sid, sid))

    if fmt == "json":
        data = []
        for r in rows:
            st = r.get("state") or {}
            data.append({"student_id": nm(r["student_id"]), "session": r["session"],
                         "topic": r.get("topic"), "transcript": st.get("transcript", [])})
        body = json.dumps({"exported": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "scope": scope, "anonymized": anon, "count": len(data), "sessions": data},
                          ensure_ascii=False, indent=2).encode("utf-8")
        return body, "application/json; charset=utf-8", f"로그원본_{scope}_{tag}_{stamp}.json"
    if fmt == "summary":
        buf = io.StringIO(); buf.write("﻿"); w = _csv.writer(buf)
        w.writerow(["학생","차시","주제","턴수","요소누적도","대화유형","자발비율","마지막발화","수준","사유"])
        for s in sums:
            w.writerow([nm(s.get("student_id")), s.get("session"), s.get("topic"), s.get("turns"),
                        s.get("요소누적도"), s.get("대화유형"), s.get("자발비율"),
                        s.get("last"), s.get("level"), s.get("reason")])
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"학생별요약_{scope}_{tag}_{stamp}.csv"
    if fmt == "template":
        # 사람 코딩용 빈 서식 — 학생/AI 발화만 채워주고 코드 칸은 비워 둔다.
        # 코더 신뢰도 검증(F1·κ) 때 이 서식을 사람이 채운 뒤 validate.py로 기계 코딩과 대조한다.
        buf = io.StringIO(); buf.write("﻿"); w = _csv.writer(buf)
        w.writerow(["학생","차시","턴","학생발화","AI응답",
                    "상황","조건","목적","대상","계기","개입","행위유형"])
        for r in rows:
            st = r.get("state") or {}
            for t in st.get("transcript", []):
                w.writerow([nm(r["student_id"]), r["session"], t.get("turn"),
                            t.get("student") or "", t.get("ai") or "",
                            "","","","","","",""])   # 사람이 채울 칸(빈칸)
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"사람코딩서식_{scope}_{tag}_{stamp}.csv"
    # 기본: 턴별 코딩(기계) — 사람코딩서식과 열 이름이 같아 validate.py로 바로 대조 가능
    buf = io.StringIO(); buf.write("﻿"); w = _csv.writer(buf)
    w.writerow(["학생","차시","턴","시각","학생발화","AI응답",
                "상황","조건","목적","대상","계기","개입","행위유형","안전OK","위험범주"])
    for r in rows:
        st = r.get("state") or {}
        for t in st.get("transcript", []):
            # 이번 턴 탐지 요소 우선, 구(舊)데이터는 누적 요소로 폴백
            e = t.get("turn_elements") or t.get("elements", {}) or {}
            w.writerow([nm(r["student_id"]), r["session"], t.get("turn"), t.get("ts") or "",
                        t.get("student") or "", t.get("ai") or "",
                        int(bool(e.get("상황"))), int(bool(e.get("조건"))),
                        int(bool(e.get("목적"))), int(bool(e.get("대상"))),
                        t.get("계기"), t.get("개입"), t.get("행위유형") or "",
                        t.get("safety_ok"), t.get("category") or ""])
    return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"턴별코딩_{scope}_{tag}_{stamp}.csv"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- 인증 헬퍼 ---------------------------------------------------------
    def _cookie(self, name):
        for part in (self.headers.get("Cookie", "") or "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                if k == name:
                    return v
        return None

    def _role(self):
        """쿠키의 접근 코드로 역할 판정: admin / teacher / None."""
        v = self._cookie("pgauth")
        if v and v == ADMIN_KEY:
            return "admin"
        if v and v == TEACHER_KEY:
            return "teacher"
        return None

    def _lesson_from(self, path, prefix):
        """/prefix/<n> 에서 차시 번호(1~4)를 뽑는다. 없으면 기본 차시."""
        tail = path[len(prefix):].strip("/")
        try:
            n = int(tail.split("/")[0])
            return n if 1 <= n <= 4 else LESSON_SESSION
        except Exception:
            return LESSON_SESSION

    def do_GET(self):
        path = urlparse(self.path).path
        role = self._role()

        # 관리자 허브(나만) — 1~4차시 전체
        if path in ("/", "/admin", "/index.html"):
            if role == "admin":
                return self._serve_templated("admin.html", {})
            return self._serve_login("admin")

        # 학생 화면 — /s/<n> (코드 불필요, 해당 차시 고정)
        if path == "/s" or path.startswith("/s/"):
            n = self._lesson_from(path, "/s")
            return self._serve_templated("index.html", {"__LESSON__": str(n)})

        # 교사 대시보드 — /teacher/<n> (교사/관리자 코드 필요, 그 차시만)
        if path == "/teacher" or path.startswith("/teacher/"):
            if role in ("admin", "teacher"):
                n = self._lesson_from(path, "/teacher")
                return self._serve_templated("teacher.html", {"__SCOPE__": str(n)})
            return self._serve_login("teacher")

        if path == "/api/dashboard":
            if role not in ("admin", "teacher"):
                return self._send(403, {"error": "권한이 필요합니다."})
            q = parse_qs(urlparse(self.path).query)
            only = q.get("session", [None])[0]
            return self._send(200, {"students": dashboard_data(only)})
        if path == "/api/transcript":
            if role not in ("admin", "teacher"):
                return self._send(403, {"error": "권한이 필요합니다."})
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("student") or [""])[0]
            session = int((q.get("session") or ["1"])[0])
            S = _lookup_session(sid, session)     # 캐시 없으면 store에서 복원(재시작 후에도 보임)
            return self._send(200, {"student": sid, "session": session,
                                    "turns": S["transcript"] if S else []})
        if path == "/api/pending":               # 학생 폴링 — 코드 불필요
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("student") or [""])[0]
            session = int((q.get("session") or ["1"])[0])
            token = (q.get("token") or [""])[0]
            if not sid:
                return self._send(200, {"reply": None})
            return self._send(200, handle_pending(sid, session, token))
        if path == "/api/direct":                # 1:1 직접 대화 조회 — 학생(본인)·교사 공용
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("student") or [""])[0]
            session = int((q.get("session") or ["1"])[0])
            since = int((q.get("since") or ["0"])[0])
            if not sid:
                return self._send(200, {"direct": [], "total": 0, "call": False})
            return self._send(200, handle_direct(sid, session, since))
        if path == "/api/health":                 # 교사/관리자 — 서비스 상태(품질저하·정지)
            if role not in ("admin", "teacher"):
                return self._send(403, {"error": "권한이 필요합니다."})
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            return self._send(200, {"mode": "live" if has_key else "mock",
                                    "degraded": bool(llm.degraded_recently()),
                                    "paused": PAUSED})
        if path == "/api/export":                # 관리자 전용 로그 내보내기
            if role != "admin":
                return self._send(403, {"error": "관리자만 내보낼 수 있습니다."})
            q = parse_qs(urlparse(self.path).query)
            fmt = (q.get("fmt") or ["turns"])[0]
            sraw = (q.get("session") or ["all"])[0]
            only = None if sraw in ("all", "", None) else int(sraw)
            anon = (q.get("anon") or ["1"])[0] != "0"   # 기본 익명화, anon=0이면 실명 포함
            body, ctype, fname = _export_bytes(fmt, only, anon=anon)
            from urllib.parse import quote
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition",
                             "attachment; filename*=UTF-8''" + quote(fname))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if path == "/api/whoami":
            return self._send(200, {"role": role or "none"})
        if path == "/logout":
            self.send_response(302)
            self.send_header("Set-Cookie", "pgauth=; Path=/; Max-Age=0")
            self.send_header("Location", "/")
            self.end_headers(); return
        if path.startswith("/static/"):
            return self._serve_file(os.path.join(STATIC, path[len("/static/"):]),
                                    "text/html; charset=utf-8")
        self._send(404, {"error": "not found"})

    def _serve_templated(self, name, repl):
        fp = os.path.join(STATIC, name)
        if not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        html = open(fp, encoding="utf-8").read()
        for k, v in repl.items():
            html = html.replace(k, v)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_login(self, want):
        """want: 'admin' | 'teacher' — 로그인 화면(접속 코드 입력)."""
        self._serve_templated("login.html", {"__WANT__": want})

    def _serve_file(self, fp, ctype):
        if not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        try:
            if path == "/api/login":                      # 접속 코드 확인 → 쿠키 발급
                key = (body.get("key") or "").strip()
                role = "admin" if key == ADMIN_KEY else ("teacher" if key == TEACHER_KEY else None)
                if not role:
                    return self._send(200, {"ok": False, "error": "코드가 올바르지 않습니다."})
                data = json.dumps({"ok": True, "role": role}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                # 세션 쿠키(브라우저 닫으면 만료). 운영 시 HTTPS+Secure 권장.
                self.send_header("Set-Cookie", f"pgauth={key}; Path=/; HttpOnly; SameSite=Lax")
                self.end_headers(); self.wfile.write(data); return
            if path == "/api/join":                        # 학생 입장 — 동일 이름 이어받기
                return self._send(200, handle_join(body))
            if path == "/api/turn":                        # 학생: 코드 불필요
                return self._send(200, handle_turn(body))
            if path == "/api/call":                        # 학생: 선생님 호출벨
                sid = (body.get("student") or "").strip()
                sess = int(body.get("session", 1))
                if not sid: return self._send(200, {"error": "이름 없음"})
                return self._send(200, handle_call(sid, sess))
            if path == "/api/direct_send":                 # 학생: 1:1 직접 대화에 보내기
                sid = (body.get("student") or "").strip()
                sess = int(body.get("session", 1))
                if not sid: return self._send(200, {"error": "이름 없음"})
                return self._send(200, handle_direct_send(sid, sess, body.get("text"), "student"))

            # 아래는 교사/관리자만
            role = self._role()
            if role not in ("admin", "teacher"):
                return self._send(403, {"error": "권한이 필요합니다."})
            if path == "/api/ack":                         # 교사: 알림 확인 처리
                sid = body.get("student_id"); sess = int(body.get("session", 1))
                S = _get_session(sid, sess, "자유")
                S["ack"] = True
                _save_session(sid, sess, S)                 # 확인 상태도 영속화(레벨 재계산)
                return self._send(200, {"ok": True})
            if path == "/api/teacher_reply":               # 교사: 학생에게 직접 답 보내기
                sid = body.get("student_id"); sess = int(body.get("session", 1))
                return self._send(200, handle_teacher_reply(sid, sess, body.get("text")))
            if path == "/api/allow_ai":                    # 교사: 사소한 위험 통과 → AI가 답하게
                sid = body.get("student_id"); sess = int(body.get("session", 1))
                return self._send(200, handle_allow_ai(sid, sess))
            if path == "/api/direct_reply":                # 교사: 1:1 직접 대화에 답하기(호출 해제)
                sid = body.get("student_id"); sess = int(body.get("session", 1))
                return self._send(200, handle_direct_send(sid, sess, body.get("text"), "teacher"))
            if path == "/api/broadcast":                   # 교사/관리자: 전체 학생에게 공지
                sraw = body.get("session")
                only = None if sraw in (None, "", "all") else int(sraw)
                return self._send(200, handle_broadcast(body.get("text"), only))
            if path == "/api/pause":                       # 관리자/교사: 전역 비상 정지 토글
                global PAUSED
                PAUSED = bool(body.get("on"))
                return self._send(200, {"ok": True, "paused": PAUSED})
            if path == "/api/reset":                        # 교사: 실습 초기화(해당 차시만)
                sess = body.get("session")
                if sess is None:
                    SESSIONS.clear()
                else:
                    for k in [k for k in SESSIONS if k[1] == int(sess)]:
                        del SESSIONS[k]
                store.clear(None if sess is None else int(sess))  # 영속본도 초기화
                return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):  # 조용히
        pass


def _lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


store.init()   # 저장소 준비(테이블 생성 등). DATABASE_URL 없으면 메모리 폴백.

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mode = "실 LLM(" + os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001") + ")" \
        if os.environ.get("ANTHROPIC_API_KEY") else "MOCK(키 없음)"
    ip = _lan_ip()
    print("=" * 60)
    print(f"  prompt-gate 샘플 서버  ·  {mode} 모드")
    print(f"  관리자 허브(나만):  http://localhost:{port}/           코드: ADMIN_KEY")
    print(f"  학생용(차시별):     http://localhost:{port}/s/1  …  /s/4")
    print(f"  교사 대시보드:      http://localhost:{port}/teacher/1  (코드: TEACHER_KEY)")
    print(f"  같은 와이파이 IP:   http://{ip}:{port}      (학생 기기에서 /s/1 로 접속)")
    print(f"  턴 로그:            {LOGCSV}")
    print(f"  상태 저장소:        {store.backend()}  (postgres면 재시작 후 이어가기 O)")
    print("=" * 60)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
