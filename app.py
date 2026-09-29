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
import os, json, csv, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from detector import ConversationCoder, NudgePolicy, LOG_FIELDS
import llm

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

# (student_id, session) -> {coder, messages}
SESSIONS = {}


def _get_session(sid, session, topic):
    key = (sid, int(session))
    if key not in SESSIONS:
        SESSIONS[key] = {
            "coder": ConversationCoder(sid, int(session), topic or "자유",
                                       policy=NudgePolicy(session=int(session))),
            "messages": [], "updated": time.time(), "ack": True, "last": "",
            "last_elicited": False, "transcript": [],
        }
    return SESSIONS[key]


def _append_transcript(entry: dict):
    try:
        with open(TRANSCRIPT, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _asks(text: str) -> bool:
    """AI 응답이 되묻기(유도)를 포함했는지 간단 판정."""
    return bool(text) and ("?" in text or "까" in text[-3:] or "래" in text[-3:])


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
    rows = []
    for (sid, session), S in SESSIONS.items():
        if only_session is not None and session != int(only_session):
            continue
        coder = S["coder"]; s = coder.summary()
        level, reason = compute_status(S)
        rows.append({
            "student_id": sid, "session": session, "topic": coder.meta[2],
            "turns": s["총턴수"], "요소누적도": s["요소누적도"],
            "대화유형": s["대화유형"], "자발비율": s["자발비율"],
            "last": (S.get("last") or "")[:30], "level": level, "reason": reason,
            "ago": int(time.time() - S["updated"]),
        })
    order = {"즉시": 0, "주의": 1, "로그": 2}
    rows.sort(key=lambda r: (order[r["level"]], r["ago"]))
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

    S = _get_session(sid, session, topic)

    # 연속 전송 방지: 마지막 전송 후 COOLDOWN_SEC 이내면 처리하지 않음(서버측 강제)
    now = time.time()
    since = now - S.get("last_turn_ts", 0)
    if since < COOLDOWN_SEC:
        remain = int(COOLDOWN_SEC - since) + 1
        return {"error": f"조금만 천천히! {remain}초 뒤에 다시 보내요.", "cooldown": True}
    S["last_turn_ts"] = now

    coder, messages = S["coder"], S["messages"]

    # 직전 AI가 되물었는지를 이번 턴 계기 판정에 사용
    res = coder.step(utterance, elicited_prev=S["last_elicited"])
    from dataclasses import asdict
    _append_log(asdict(res["log"]))      # 코딩 시트 스키마로 적재

    messages.append({"role": "user", "content": utterance})
    S["updated"] = time.time()
    S["last"] = utterance

    reply = None
    if res["safety"].ok:
        filled = sum(1 for v in coder.elements().values() if v)
        reply = llm.chat(messages, filled=filled, session=int(session))
        messages.append({"role": "assistant", "content": reply})
        S["last_elicited"] = _asks(reply)     # 이번 AI 응답이 되물었는가
    else:
        S["ack"] = False                      # 새 안전 알림 → 미확인
        S["last_elicited"] = False

    L = res["log"]
    entry = {"turn": L.턴, "student": utterance, "ai": reply,
             "계기": L.계기, "개입": L.개입,
             "safety_ok": res["safety"].ok, "category": res["safety"].category,
             "elements": coder.elements()}
    S["transcript"].append(entry)
    _append_transcript({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "student_id": sid, "session": int(session), **entry})

    return {
        "reply": reply,
        "elements": coder.elements(),
        "safety": {"ok": res["safety"].ok, "category": res["safety"].category},
        "summary": coder.summary(),
    }


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
            S = SESSIONS.get((sid, session))
            return self._send(200, {"student": sid, "session": session,
                                    "turns": S["transcript"] if S else []})
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
            if path == "/api/turn":                        # 학생: 코드 불필요
                return self._send(200, handle_turn(body))

            # 아래는 교사/관리자만
            role = self._role()
            if role not in ("admin", "teacher"):
                return self._send(403, {"error": "권한이 필요합니다."})
            if path == "/api/ack":                         # 교사: 알림 확인 처리
                key = (body.get("student_id"), int(body.get("session", 1)))
                if key in SESSIONS:
                    SESSIONS[key]["ack"] = True
                return self._send(200, {"ok": True})
            if path == "/api/reset":                        # 교사: 실습 초기화(해당 차시만)
                sess = body.get("session")
                if sess is None:
                    SESSIONS.clear()
                else:
                    for k in [k for k in SESSIONS if k[1] == int(sess)]:
                        del SESSIONS[k]
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
    print("=" * 60)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
