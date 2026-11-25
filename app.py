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
import roster
import transfer
import coding

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

# ---- 본실험(2026-12) 되묻기 규칙 — 시스템이 지킨다 -------------------------------
# 2차시 점검 요청(조건 check2가 켜진 학생만): 답 끝에 아래 문장 중 하나를 차례로 붙인다.
CHECK2_PHRASES = [
    "이 답이 네가 찾던 거 맞아?",
    "이대로 괜찮아? 더 필요한 게 있으면 말해줘.",
    "혹시 더 알려주고 싶은 게 있어?",
]
# 3차시 점검 한마디: 시스템이 '막힘'을 판정했을 때만 붙인다.
CHECK3_PHRASE = "보내기 전에 스스로 한번 점검해봤어?"
# 막힘 판정: 지금 이야기에서 빠진 요소가 남아 있는데, 새 요소 없이 짧은 말(STUCK_SHORT_CHARS자 미만)이
# STUCK_RUN번 이어지면 막힘. 한 번 점검 한마디가 나간 뒤에는 다시 STUCK_RUN번이 쌓여야 한다.
STUCK_RUN = int(os.environ.get("STUCK_RUN", 3))
STUCK_SHORT_CHARS = int(os.environ.get("STUCK_SHORT_CHARS", 15))
# 2~4차시 답에 질문이 섞이면 다시 생성하는 최대 횟수. 그래도 남으면 질문 문장만 지운다.
MAX_REGEN = int(os.environ.get("MAX_REGEN", 2))


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
    session = int(body.get("session", 1))
    token = (body.get("token") or "").strip()
    if roster.has_classes():
        # 고정 학생 계정 — 학급·번호(·입장 번호)로만 들어온다. 서버에는 가명 코드(A-07)만 남는다.
        # (새로고침 때는 화면이 기억한 가명 코드로 다시 들어온다.)
        if body.get("sid") and roster.is_roster_sid(body["sid"]):
            sid = body["sid"]
        else:
            r = roster.resolve_login(body.get("cls"), body.get("num"), body.get("pin", ""))
            if not r["ok"]:
                return r
            sid = r["sid"]
    else:
        # 명단이 없을 때(시연·연수용): 예전처럼 이름으로 들어온다.
        sid = (body.get("student") or "").strip()
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
    dim = {e: False for e in ELEMENTS}
    if S:
        for t in S["transcript"]:
            hist.append({"student": t.get("student"), "ai": t.get("ai"),
                         "pending": bool(t.get("pending"))})
        elems = S["coder"].elements()
        dim = S["coder"].dim()
        direct = list(S.get("direct", []))
    cond = roster.condition_for(sid)
    if not cond["lights"]:                     # 표시등 끈 조건: 화면에 요소 상태를 보내지 않음
        elems, dim = {}, {}
    return {"ok": True, "sid": sid, "token": token, "took_over": took_over,
            "lights": cond["lights"],
            "history": hist, "elements": elems, "dim": dim, "direct": direct}


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
        "check2_i": 0,         # 2차시 점검 요청 문장 순번(차례로 돌림)
        "stuck_last": 0,       # 3차시 점검 한마디가 마지막으로 나간 턴
    }


def _serialize(S) -> dict:
    """세션 상태를 저장 가능한 dict로. 대화 복원에 필요한 코더 내부까지 포함."""
    from dataclasses import asdict
    c = S["coder"]
    return {
        "coder": {"history": c._history, "cum": c._cum, "turn": c._turn,
                  "cum_session": c._cum_session, "topic_no": c._topic_no,
                  "topic_start": c._topic_start,
                  "logs": [asdict(l) for l in c.logs]},
        "messages": S["messages"], "ack": S["ack"], "last": S["last"],
        "last_elicited": S["last_elicited"], "updated": S["updated"],
        "last_turn_ts": S.get("last_turn_ts", 0), "transcript": S["transcript"],
        "outbox": S.get("outbox", []), "direct": S.get("direct", []),
        "call": S.get("call", False), "call_ts": S.get("call_ts", 0),
        "check2_i": S.get("check2_i", 0), "stuck_last": S.get("stuck_last", 0),
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
    cs = cd.get("cum_session", cum)            # 이전 저장본엔 없음 → 기존 누적으로 대체
    c._cum_session = {e: bool(cs.get(e, False)) or c._cum[e] for e in ELEMENTS}
    c._topic_no = int(cd.get("topic_no", 1))
    c._topic_start = int(cd.get("topic_start", 0))
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
    S["check2_i"] = int(state.get("check2_i", 0))
    S["stuck_last"] = int(state.get("stuck_last", 0))
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
        "cond": _cond_short(roster.condition_for(sid)),
    }


def _cond_short(c: dict) -> str:
    """대시보드용 짧은 조건 표시. 예: '표시등○ 점검○'(켬) / '표시등× 점검○'."""
    if c.get("support") == "always":
        s = f"계속되묻기 표시등{'○' if c['lights'] else '×'}"
    else:
        s = f"표시등{'○' if c['lights'] else '×'} 점검{'○' if c['check2'] else '×'}"
    return s + ("" if c["assigned"] else " (미배정)")


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


_SENT_SPLIT = re.compile(r'(?<=[.!?…~])\s+|\n+')


def _strip_questions(text: str) -> str:
    """질문(되묻기·유도)으로 판정되는 문장만 지운다. 다시 생성해도 질문이 남을 때의 마지막 방어."""
    sents = [s for s in _SENT_SPLIT.split(text or "") if s.strip()]
    keep = [s for s in sents if not _asks(s)]
    return " ".join(keep).strip()


def _is_stuck(S) -> bool:
    """3차시 '막힘' 판정 — 지금 이야기에 빠진 요소가 남아 있고, 최근 STUCK_RUN턴이 모두
       '새 요소 없음 + 짧은 말'이며, 직전 점검 한마디 뒤로 STUCK_RUN턴 이상 지났을 때."""
    coder = S["coder"]
    logs = coder.logs
    if len(logs) < STUCK_RUN:
        return False
    if all(coder.elements().values()):              # 지금 이야기의 네 요소를 다 갖췄으면 막힘 아님
        return False
    if coder._turn - S.get("stuck_last", 0) < STUCK_RUN:
        return False
    recent = logs[-STUCK_RUN:]
    return all(l.주제내신규요소수 == 0 and len((l.학생발화 or "").strip()) < STUCK_SHORT_CHARS
               for l in recent)


def _generate_reply(S, session: int, cond: dict) -> tuple:
    """AI 답 만들기 — 차시 규칙과 조건을 시스템이 지킨다. (reply, gen) 반환.
       gen은 턴 로그에 그대로 남는다(논문의 조작 점검 근거)."""
    session = int(session)
    filled = sum(1 for v in S["coder"].elements().values() if v)
    sup = cond.get("support", "fade")
    ans_only = llm.answer_only(session, sup)
    gen = {"방식": ("계속되묻기" if sup == "always" else ("답만" if ans_only else "되묻기")),
           "다시생성": 0, "질문막기": "", "덧붙임": "", "막힘": False}
    reply = llm.chat(S["messages"], filled=filled, session=session, support=sup)
    if ans_only:
        # ① 2~4차시: 답에 질문이 섞이면 더 강한 지시로 다시 생성
        while _asks(reply) and gen["다시생성"] < MAX_REGEN:
            gen["다시생성"] += 1
            reply = llm.chat(S["messages"], filled=filled, session=session, strict=True, support=sup)
        if _asks(reply):
            stripped = _strip_questions(reply)
            if stripped:
                reply, gen["질문막기"] = stripped, "질문문장삭제"
            else:
                gen["질문막기"] = "실패"            # 지울 수 없음 → 그대로 두고 기록
        else:
            gen["질문막기"] = "다시생성" if gen["다시생성"] else "통과"
        # ② 시스템이 붙이는 점검 한마디
        if session == 2 and cond.get("check2"):
            i = S.get("check2_i", 0)
            gen["덧붙임"] = CHECK2_PHRASES[i % len(CHECK2_PHRASES)]
            S["check2_i"] = i + 1
        elif session == 3 and _is_stuck(S):
            gen["덧붙임"] = CHECK3_PHRASE
            gen["막힘"] = True
            S["stuck_last"] = S["coder"]._turn
        if gen["덧붙임"]:
            reply = (reply.rstrip() + " " + gen["덧붙임"]).strip()
    return reply, gen


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
        r["cond"] = _cond_short(roster.condition_for(r.get("student_id", "")))   # 늘 현재 조건으로
    order = {"즉시": 0, "주의": 1, "로그": 2}
    rows.sort(key=lambda r: (order.get(r.get("level", "로그"), 2), r["ago"]))
    return rows


def _append_log(row: dict):
    from dataclasses import asdict
    new = not os.path.exists(LOGCSV)
    if not new:   # 열이 바뀐 이전 로그(주제번호·주제전환 없음)는 따로 보관하고 새 파일로 시작
        with open(LOGCSV, encoding="utf-8-sig") as f:
            head = f.readline().strip().split(",")
        if head != LOG_FIELDS:
            os.replace(LOGCSV, LOGCSV.replace(".csv", time.strftime("_old_%Y%m%d%H%M%S.csv")))
            new = True
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

    # 명단을 쓰는 동안에는 명단에 있는 가명 코드로만 대화한다(이름을 바꿔 새 학생이 생기는 것 방지)
    if roster.has_classes() and not roster.is_roster_sid(sid):
        return {"error": "다시 들어와줘. (화면을 새로고침하고 반과 번호를 골라줘)", "relogin": True}

    # 다른 기기가 같은 이름으로 이어받았으면 이 탭은 종료
    if _is_kicked(sid, session, (body.get("token") or "").strip()):
        return {"kicked": True}

    S = _get_session(sid, session, topic)
    cond = roster.condition_for(sid)            # 이 학생의 실험 조건(매 턴 로그에 남김)
    _touch_presence(sid, session)   # 이 이름이 아직 접속 중임을 갱신

    # 관리자 비상 정지 — 새 턴을 처리하지 않고 안내만
    if PAUSED:
        return {"reply": "지금은 잠깐 쉬는 시간이야. 선생님 안내를 기다려줘!",
                **_lights_payload(S, cond), "paused": True}
    # 학생당 턴 상한 — 비용 폭주·도배 방지(초과 시 LLM 호출 없이 부드럽게 멈춤)
    if S["coder"]._turn >= MAX_STUDENT_TURNS:
        return {"reply": "오늘 정말 많이 이야기했다! 잠깐 쉬었다가 선생님이 안내해줄 거야.",
                **_lights_payload(S, cond), "capped": True}

    # 연속 전송 방지: 마지막 전송 후 COOLDOWN_SEC 이내면 처리하지 않음(서버측 강제)
    now = time.time()
    since = now - S.get("last_turn_ts", 0)
    if since < COOLDOWN_SEC:
        remain = int(COOLDOWN_SEC - since) + 1
        return {"error": f"조금만 천천히! {remain}초 뒤에 다시 보내요.", "cooldown": True}
    S["last_turn_ts"] = now

    coder, messages = S["coder"], S["messages"]

    # 학생이 이 말을 쓰는 동안 보고 있던 표시등 상태(진하게·흐리게·없음). 표시등 끈 조건은 '꺼짐'.
    _on, _dim = coder.elements(), coder.dim()
    lights_before = {e: ("진하게" if _on[e] else "흐리게" if _dim[e] else "없음") for e in ELEMENTS}
    if not cond["lights"]:
        lights_before = {e: "꺼짐" for e in ELEMENTS}
    ai_asked_prev = bool(S["last_elicited"])
    meta = _clean_meta(body.get("meta") or {}, utterance)

    # 직전 AI가 되물었는지를 이번 턴 계기 판정에 사용
    # (coder.step 안에서 Layer1 키워드 + Layer3 LLM 2차로 입력 안전을 점검)
    res = coder.step(utterance, elicited_prev=S["last_elicited"])
    from dataclasses import asdict
    L = res["log"]

    messages.append({"role": "user", "content": utterance})
    S["updated"] = time.time()
    S["last"] = utterance

    reply = None
    gen = {}
    if res["safety"].ok:
        reply, gen = _generate_reply(S, int(session), cond)   # 차시 규칙·조건을 시스템이 지킴
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
             "elements_session": coder.elements_session(), "dim": coder.dim(),
             "topic_no": L.주제번호, "topic_shift": bool(L.주제전환),
             "ts": time.strftime("%Y-%m-%d %H:%M:%S"),    # 턴별 시각(프로세스 분석용)
             # ---- 본실험 기록 보강 ----
             "cond": {"lights": cond["lights"], "check2": cond["check2"],
                      "support": cond.get("support", "fade"),
                      "unit": cond["unit"], "assigned": cond["assigned"]},
             "lights_before": lights_before,             # 이 말을 쓰는 동안 본 표시등
             "ai_asked_prev": ai_asked_prev,             # 직전 AI 답이 되묻기였나
             "ai_asks": bool(reply) and _asks(reply),    # 이번 AI 답(최종)이 되묻기인가
             "gen": gen,                                  # 생성 방식·다시 생성·덧붙인 점검·막힘
             "new_topic_elems": L.주제내신규요소수,
             **meta}                                      # 예시 버튼·고친 횟수·쓰는 시간
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
        return {"status": "waiting", **_lights_payload(S, cond, L),
                "care": (res["safety"].category == CARE_CATEGORY)}
    return {"reply": reply, **_lights_payload(S, cond, L)}


def _lights_payload(S, cond, L=None) -> dict:
    """학생 화면에 보낼 표시등 정보. 표시등 끈 조건이면 아무것도 보내지 않는다
       (요소 판정은 서버에서 그대로 돌고 로그에 남는다)."""
    if not cond["lights"]:
        return {"lights": False}
    out = {"lights": True, "elements": S["coder"].elements(), "dim": S["coder"].dim()}
    if L is not None:
        out["topic_shift"] = bool(L.주제전환)
    return out


def _clean_meta(m: dict, utterance: str) -> dict:
    """학생 화면이 보낸 입력 기록(예시 버튼·고친 횟수·쓰는 시간)을 검증해 정리."""
    def _int(v, hi):
        try:
            return max(0, min(int(v), hi))
        except Exception:
            return None
    seed = (m.get("seed_text") or "")[:100] if m.get("seed_used") else ""
    return {
        "seed_used": bool(m.get("seed_used")),            # 이번 말을 쓰는 동안 예시 버튼을 눌렀나
        "seed_text": seed,                                # 누른 예시 문구
        "seed_exact": bool(seed) and utterance.strip() == seed.strip(),   # 예시를 그대로 보냈나
        "revisions": _int(m.get("revisions"), 10000),     # 보내기 전에 지운 횟수(연속으로 지운 것은 1번)
        "deleted_chars": _int(m.get("deleted_chars"), 100000),
        "compose_ms": _int(m.get("compose_ms"), 3600_000),  # 처음 쓰기 시작해서 보내기까지 걸린 시간
        "inputs": _int(m.get("inputs"), 100000),          # 입력 이벤트 수
        "pastes": _int(m.get("pastes"), 1000),            # 붙여넣기 횟수
    }


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


def _deliver_reply(sid, session, reply_text, kind, gen=None):
    """교사 답변/허용을 세션에 반영하고, 학생 폴링(/api/pending)으로 전달할 슬롯에 넣는다."""
    S = _get_session(sid, session, "자유")
    tr = S["transcript"]
    target = next((e for e in reversed(tr) if e.get("pending")), None)
    if target is not None:                      # 대기 중이던 차단 턴을 교사 답으로 채움
        target["ai"] = reply_text; target["pending"] = False; target["개입"] = kind
        target["ai_asks"] = _asks(reply_text)
        if gen is not None:
            target["gen"] = gen
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
    # 교사 승인 → 안전 재검사 생략. 차시 규칙·조건은 평소와 똑같이 시스템이 지킨다.
    reply, gen = _generate_reply(S, int(session), roster.condition_for(sid))
    _deliver_reply(sid, session, reply, "교사허용", gen=gen)
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
        return {"reply": reply, **_lights_payload(S, roster.condition_for(sid))}
    return {"reply": None}


def handle_direct(sid, session, since=0):
    """1:1 직접 대화 조회 — since 이후의 메시지와 호출 상태를 돌려준다(학생·교사 공용)."""
    S = _lookup_session(sid, session)
    if not S:
        return {"direct": [], "total": 0, "call": False}
    d = S.get("direct") or []
    return {"direct": d[since:], "total": len(d), "call": bool(S.get("call"))}


DIRECT_COOLDOWN = int(os.environ.get("DIRECT_COOLDOWN", 2))  # 학생측 호출·직접전송 연타 방지(초)
MAX_DIRECT = 100                                             # 직접대화 보관 최대 개수(무한증가 방지)


def handle_call(sid, session):
    """학생이 '선생님 부르기'를 누름 → 호출 플래그 ON(대시보드에 표시). 연타는 무시."""
    S = _get_session(sid, session, "자유")
    now = time.time()
    if now - S.get("last_direct_ts", 0) < DIRECT_COOLDOWN:
        return {"ok": True, "throttled": True}
    S["last_direct_ts"] = now
    S["call"] = True; S["call_ts"] = now
    _save_session(sid, session, S)
    return {"ok": True}


def handle_direct_send(sid, session, text, who):
    """1:1 직접 대화에 메시지 추가. who='student'|'teacher'. 교사가 답하면 호출 해제.
       학생측은 연타 쿨다운, 기록은 최근 MAX_DIRECT개로 제한."""
    text = (text or "").strip()
    if not text:
        return {"error": "빈 메시지"}
    S = _get_session(sid, session, "자유")
    now = time.time()
    if who == "student" and now - S.get("last_direct_ts", 0) < DIRECT_COOLDOWN:
        return {"ok": True, "throttled": True}       # 도배 방지 — 조용히 무시
    d = S.setdefault("direct", [])
    d.append({"from": who, "text": text[:500], "ts": time.strftime("%H:%M")})
    if len(d) > MAX_DIRECT:
        del d[:len(d) - MAX_DIRECT]                  # 오래된 것부터 잘라 최근 100개만 유지
    if who == "student":
        S["call"] = True; S["call_ts"] = now; S["last_direct_ts"] = now
    else:
        S["call"] = False                            # 교사가 응답하면 호출 해제
    S["updated"] = now
    _save_session(sid, session, S)
    return {"ok": True, "direct_total": len(d)}


def handle_broadcast(text, only_session=None):
    """전체 공지 — 접속 중(메모리 캐시)인 학생의 outbox에 즉시 넣는다.
       DB를 학생 수만큼 두드리지 않아 빠르고 안전(hang 방지). 접속 중이 아닌 학생은
       어차피 폴링을 안 하므로 대상에서 제외된다."""
    text = (text or "").strip()
    if not text:
        return {"error": "빈 공지"}
    msg = "📢 [선생님 공지] " + text
    n = 0
    for (sid, sess), S in list(SESSIONS.items()):
        if only_session is not None and sess != int(only_session):
            continue
        S.setdefault("outbox", []).append(msg)
        n += 1
    return {"ok": True, "count": n}


def handle_delete_student(sid, session):
    """특정 학생(한 명·한 차시)의 대화 전체 삭제 — 캐시·접속상태·영속본 모두 제거."""
    sid = (sid or "").strip()
    if not sid:
        return {"error": "학생을 지정해주세요."}
    sess = int(session)
    SESSIONS.pop((sid, sess), None)      # 실행 중 캐시
    PRESENCE.pop((sid, sess), None)      # 접속 상태
    store.delete(sid, sess)              # 영속본(Neon/메모리)
    return {"ok": True}


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
    # 명단 학생(A-07 같은 가명 코드)은 이미 실명이 아니므로 그대로 둔다 → 차시·내보내기마다 같은 코드.
    ids = sorted({r["student_id"] for r in rows} | {s.get("student_id") for s in sums})
    ids = [i for i in ids if i and not roster.parse_sid(i)[0]]
    alias = {sid: f"S{i:02d}" for i, sid in enumerate(ids, 1)}
    nm = (lambda sid: sid) if not anon else (lambda sid: alias.get(sid, sid))

    if fmt == "roster":
        # 학생별 조건 배정표(입장 번호 제외) — 분석 때 턴 로그와 '학생' 열로 붙인다.
        buf = io.StringIO(); buf.write("﻿"); w = _csv.writer(buf)
        cols = ["학생", "학급", "번호", "배정단위", "표시등", "점검요청", "지원방식", "배정됨", "씨앗"]
        w.writerow(cols)
        for r in roster.export_rows():
            w.writerow([r[c] for c in cols])
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"조건배정표_{stamp}.csv"

    if fmt == "transfer":
        # 전이 과제 응답 — 조건 배정표와 '학생' 열로 붙는다. 자동 판정은 사람 코딩 비교용.
        buf = io.StringIO(); buf.write("\ufeff"); w = _csv.writer(buf)
        conds = {r["학생"]: r for r in roster.export_rows()}
        w.writerow(["학생", "학급", "지원방식", "표시등", "점검요청", "회차", "문항", "문항제목", "제출시각",
                    "응답", "글자수", "쓰기시간초", "고친횟수", "지운글자수", "붙여넣기",
                    "자동_상황", "자동_조건", "자동_목적", "자동_대상", "자동_위험"])
        for r in transfer.all_responses():
            c = conds.get(r["sid"], {})
            a = r.get("auto") or {}
            cms = r.get("compose_ms")
            w.writerow([r["sid"], c.get("학급", ""), c.get("지원방식", ""), c.get("표시등", ""),
                        c.get("점검요청", ""), transfer.WAVE_LABEL.get(r["wave"], r["wave"]), r["item"],
                        r.get("title", ""), r.get("ts", ""), r.get("text", ""), len(r.get("text") or ""),
                        "" if cms is None else round(cms / 1000, 1),
                        "" if r.get("revisions") is None else r["revisions"],
                        "" if r.get("deleted_chars") is None else r["deleted_chars"],
                        "" if r.get("pastes") is None else r["pastes"],
                        *[("" if e not in a else int(bool(a[e]))) for e in ELEMENTS],
                        "" if "risk" not in a else int(bool(a["risk"]))])
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"전이과제_{stamp}.csv"
    if fmt == "coding_sheet":
        body = coding.sheet_csv()
        return body, "text/csv; charset=utf-8", f"이중코딩_시트_{stamp}.csv"
    if fmt == "coding_key":
        body = coding.key_csv()
        return body, "text/csv; charset=utf-8", f"이중코딩_대응표_관리자보관_{stamp}.csv"
    if fmt == "coding_disagree":
        body = coding.disagree_csv()
        return body, "text/csv; charset=utf-8", f"이중코딩_불일치합의시트_{stamp}.csv"
    if fmt == "coding_final":
        body = coding.final_csv()
        return body, "text/csv; charset=utf-8", f"이중코딩_최종코드_관리자보관_{stamp}.csv"
    if fmt == "coding_report":
        body = coding.report_csv()
        return body, "text/csv; charset=utf-8", f"이중코딩_일치도_{stamp}.csv"

    if fmt == "json":
        data = []
        for r in rows:
            st = r.get("state") or {}
            data.append({"student_id": nm(r["student_id"]), "session": r["session"],
                         "topic": r.get("topic"), "transcript": st.get("transcript", [])})
        body = json.dumps({"exported": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "scope": scope, "anonymized": anon, "count": len(data),
                           "conditions": roster.export_rows(),          # 학생별 조건 배정(입장 번호 제외)
                           "assignment_log": roster.admin_view()["audit"],
                           "transfer": transfer.all_responses(),       # 전이 과제 응답(조건은 conditions와 학생으로 연결)
                           "sessions": data},
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
    # 앞 15열은 예전과 같고(validate.py·기존 분석 스크립트 호환), 본실험 기록은 그 뒤에 붙는다.
    w.writerow(["학생","차시","턴","시각","학생발화","AI응답",
                "상황","조건","목적","대상","계기","개입","행위유형","안전OK","위험범주",
                "학급","배정단위","조건_표시등","조건_점검요청","조건배정됨",
                "주제번호","주제전환","주제내신규요소수",
                "직전AI되묻기","AI되묻기",
                "표시등_상황","표시등_조건","표시등_목적","표시등_대상",
                "예시버튼","예시문구","예시그대로","고친횟수","지운글자수","쓰기시간초","입력횟수","붙여넣기",
                "생성방식","다시생성","질문막기","덧붙인점검","막힘판정","조건_지원방식"])
    def _b(v):
        return "" if v is None else int(bool(v))
    for r in rows:
        st = r.get("state") or {}
        cid = roster.parse_sid(r["student_id"])[0] or ""
        for t in st.get("transcript", []):
            # 이번 턴 탐지 요소 우선, 구(舊)데이터는 누적 요소로 폴백
            e = t.get("turn_elements") or t.get("elements", {}) or {}
            c = t.get("cond") or {}
            lb = t.get("lights_before") or {}
            g = t.get("gen") or {}
            cms = t.get("compose_ms")
            w.writerow([nm(r["student_id"]), r["session"], t.get("turn"), t.get("ts") or "",
                        t.get("student") or "", t.get("ai") or "",
                        int(bool(e.get("상황"))), int(bool(e.get("조건"))),
                        int(bool(e.get("목적"))), int(bool(e.get("대상"))),
                        t.get("계기"), t.get("개입"), t.get("행위유형") or "",
                        t.get("safety_ok"), t.get("category") or "",
                        cid,
                        {"class": "학급", "student": "학생", "none": "명단밖"}.get(c.get("unit"), ""),
                        _b(c.get("lights")), _b(c.get("check2")), _b(c.get("assigned")),
                        t.get("topic_no", ""), _b(t.get("topic_shift")), t.get("new_topic_elems", ""),
                        _b(t.get("ai_asked_prev")), _b(t.get("ai_asks")),
                        lb.get("상황", ""), lb.get("조건", ""), lb.get("목적", ""), lb.get("대상", ""),
                        _b(t.get("seed_used")), t.get("seed_text") or "", _b(t.get("seed_exact")),
                        "" if t.get("revisions") is None else t.get("revisions"),
                        "" if t.get("deleted_chars") is None else t.get("deleted_chars"),
                        "" if cms is None else round(cms / 1000, 1),
                        "" if t.get("inputs") is None else t.get("inputs"),
                        "" if t.get("pastes") is None else t.get("pastes"),
                        g.get("방식", ""), g.get("다시생성", ""), g.get("질문막기", ""),
                        g.get("덧붙임", ""), _b(g.get("막힘")) if g else "",
                        {"always": "계속되묻기", "fade": "단계적감소"}.get(c.get("support"), "단계적감소" if c else "")])
    return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"턴별코딩_{scope}_{tag}_{stamp}.csv"


# ---------------------------------------------------------------------------
# 지원 없는 전이 과제
# ---------------------------------------------------------------------------
def handle_transfer_login(body: dict) -> dict:
    if not roster.has_classes():
        return {"ok": False, "error": "아직 반 명단이 없어. 선생님께 여쭤 봐."}
    if body.get("sid") and roster.is_roster_sid(body["sid"]):
        sid = body["sid"]
    else:
        r = roster.resolve_login(body.get("cls"), body.get("num"), body.get("pin", ""))
        if not r["ok"]:
            return r
        sid = r["sid"]
    return {**transfer.student_view(sid), "sid": sid}


def handle_transfer_submit(body: dict) -> dict:
    sid = (body.get("student") or "").strip()
    text = (body.get("text") or "").strip()
    meta = _clean_meta(body.get("meta") or {}, text)
    auto = {}
    if text and roster.is_roster_sid(sid):
        try:   # 사람 코딩과 비교하려고 자동 코더 판정도 남긴다(학생에게는 보이지 않음)
            from detector import detect_elements
            r = detect_elements([], text)
            auto = {e: bool(r.get(e)) for e in ELEMENTS}
            auto["risk"] = bool(r.get("risk"))
            auto["category"] = r.get("category") or ""
        except Exception as e:
            print("[transfer] 자동 판정 실패:", e)
    out = transfer.submit(sid, body.get("wave"), body.get("item"), text, meta, auto)
    if out.get("ok"):
        out.update(transfer.student_view(sid))
    return out


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

        # 지원 없는 전이 과제 — /t (AI 답·표시등 없음, 관리자가 연 회차만)
        if path in ("/t", "/t/"):
            return self._serve_templated("transfer.html", {})
        if path == "/api/transfer/view":
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("student") or [""])[0]
            if not roster.is_roster_sid(sid):
                return self._send(200, {"ok": False, "error": "반과 번호로 다시 들어와 줘."})
            return self._send(200, transfer.student_view(sid))
        if path == "/api/admin/transfer":
            if role != "admin":
                return self._send(403, {"error": "관리자만 볼 수 있습니다."})
            return self._send(200, transfer.admin_view())
        if path == "/api/admin/coding":
            if role != "admin":
                return self._send(403, {"error": "관리자만 볼 수 있습니다."})
            return self._send(200, coding.status())

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
            from detector import degraded_recently as _det_degraded
            degraded = bool(llm.degraded_recently() or _det_degraded())
            return self._send(200, {"mode": "live" if has_key else "mock",
                                    "degraded": degraded, "paused": PAUSED})
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
        if path == "/api/classes":               # 학생 입장 화면 — 반 목록·번호(조건·입장 번호는 숨김)
            return self._send(200, {"classes": roster.public_classes()})
        if path == "/api/admin/roster":          # 관리자 — 명단·입장 번호·조건·배정 기록
            if role != "admin":
                return self._send(403, {"error": "관리자만 볼 수 있습니다."})
            return self._send(200, roster.admin_view())
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
            if path == "/api/transfer/login":              # 학생: 전이 과제 입장
                return self._send(200, handle_transfer_login(body))
            if path == "/api/transfer/submit":             # 학생: 전이 과제 제출
                return self._send(200, handle_transfer_submit(body))
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

            # ---- 학급 명단·실험 조건 (관리자만) ----
            if path.startswith("/api/admin/"):
                if role != "admin":
                    return self._send(403, {"error": "관리자만 바꿀 수 있습니다."})
                act = path[len("/api/admin/"):]
                b = body
                def _opt_bool(k):
                    return None if b.get(k) is None else bool(b.get(k))
                def _seed():
                    v = b.get("seed")
                    return None if v in (None, "") else int(v)
                if act == "class_create":
                    return self._send(200, roster.create_class(b.get("cid"), b.get("n"),
                                                               b.get("unit", "class"), bool(b.get("pin"))))
                if act == "class_delete":
                    return self._send(200, roster.delete_class(b.get("cid")))
                if act == "class_resize":
                    return self._send(200, roster.resize_class(b.get("cid"), b.get("n")))
                if act == "class_set":
                    return self._send(200, roster.set_class(b.get("cid"), b.get("unit"),
                                                            _opt_bool("lights"), _opt_bool("check2"),
                                                            _opt_bool("pin"), b.get("support")))
                if act == "student_set":
                    return self._send(200, roster.set_student(b.get("cid"), b.get("num"),
                                                              _opt_bool("lights"), _opt_bool("check2"),
                                                              b.get("support")))
                if act == "randomize_students":
                    return self._send(200, roster.randomize_students(b.get("cid"), _seed(),
                                                                     bool(b.get("include_always"))))
                if act == "transfer_open":
                    return self._send(200, transfer.set_open(b.get("wave"), bool(b.get("on"))))
                if act == "transfer_items":
                    return self._send(200, transfer.set_items(b.get("wave"), b.get("items")))
                if act == "coding_build":
                    return self._send(200, coding.build(b.get("n_turns", 150), _seed(),
                                                        b.get("kinds") or None))
                if act == "coding_upload":
                    return self._send(200, coding.upload(b.get("coder"), b.get("csv") or ""))
                if act == "coding_clear":
                    return self._send(200, coding.clear_codes(b.get("coder")))
                if act == "randomize_classes":
                    return self._send(200, roster.randomize_classes(b.get("cids") or None, _seed(),
                                                                    bool(b.get("include_always"))))
                return self._send(404, {"error": "not found"})
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
            if path == "/api/delete_student":              # 교사: 특정 학생 대화 삭제
                sid = body.get("student_id"); sess = int(body.get("session", 1))
                return self._send(200, handle_delete_student(sid, sess))
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
