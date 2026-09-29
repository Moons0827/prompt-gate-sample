# -*- coding: utf-8 -*-
"""
상태 영속화 저장소 (Persistence Store)
=====================================
학생별 대화 상태를 Render 컨테이너 '바깥'에 저장해, 서버가 잠들거나
재배포·재시작돼도 같은 아이디로 다시 들어오면 대화를 이어가게 한다.

  · DATABASE_URL(예: Neon Postgres) 이 있으면 → Postgres 백엔드
  · 없으면 → 메모리 백엔드(로컬 테스트/폴백). 이 경우 재시작 시 사라짐.

한 행 = 한 (학생아이디, 차시).
  state   : 대화 복원에 필요한 전체 상태(JSON)
  summary : 대시보드가 빠르게 읽는 요약(JSON, 저장 시 미리 계산)

인터페이스(백엔드 공통):
  init()                              테이블 준비
  get_state(sid, session) -> dict|None
  put_state(sid, session, topic, state, summary)
  list_summaries(only_session=None) -> list[dict]
  ack(sid, session)                   교사 확인 → summary의 level 재계산은 app에서
  clear(session=None)                 실습 초기화(차시별 또는 전체)
  backend() -> "postgres" | "memory"
"""
from __future__ import annotations
import os, json, time

_DSN = os.environ.get("DATABASE_URL", "").strip()
_USE_PG = bool(_DSN)

# ---------------------------------------------------------------------------
# 메모리 백엔드 (폴백)
# ---------------------------------------------------------------------------
_MEM: dict[tuple, dict] = {}   # (sid, session) -> {"topic","state","summary","updated"}


def _mem_init(): pass
def _mem_get(sid, session):
    row = _MEM.get((sid, int(session)))
    return json.loads(json.dumps(row["state"])) if row else None
def _mem_put(sid, session, topic, state, summary):
    _MEM[(sid, int(session))] = {"topic": topic, "state": state,
                                 "summary": summary, "updated": time.time()}
def _mem_list(only_session=None):
    out = []
    for (sid, session), row in _MEM.items():
        if only_session is not None and session != int(only_session):
            continue
        out.append(row["summary"])
    return out
def _mem_clear(session=None):
    if session is None:
        _MEM.clear()
    else:
        for k in [k for k in _MEM if k[1] == int(session)]:
            del _MEM[k]


# ---------------------------------------------------------------------------
# Postgres 백엔드 (Neon 등)
# ---------------------------------------------------------------------------
def _pg_conn():
    import psycopg2
    # Neon 무료 컴퓨트는 유휴 시 잠들어 첫 연결이 지연될 수 있음 → 짧게 재시도
    last = None
    for _ in range(3):
        try:
            return psycopg2.connect(_DSN, connect_timeout=10)
        except Exception as e:
            last = e; time.sleep(1.5)
    raise last


def _pg_exec(sql, params=None, fetch=None):
    """연결을 열어 실행하고 '반드시 닫는다'(연결 누수 방지 — Neon 연결 한도 보호)."""
    conn = _pg_conn()
    try:
        with conn, conn.cursor() as cur:   # with conn: 성공 시 커밋/실패 시 롤백
            cur.execute(sql, params or ())
            if fetch == "one":
                r = cur.fetchone(); return r
            if fetch == "all":
                return cur.fetchall()
            return None
    finally:
        conn.close()                       # with conn 은 닫지 않으므로 명시적으로 닫음


def _pg_init():
    _pg_exec("""
        CREATE TABLE IF NOT EXISTS pg_session (
            student_id text NOT NULL,
            session    int  NOT NULL,
            topic      text,
            state      jsonb,
            summary    jsonb,
            updated    double precision,
            PRIMARY KEY (student_id, session)
        );
    """)


def _pg_get(sid, session):
    r = _pg_exec("SELECT state FROM pg_session WHERE student_id=%s AND session=%s",
                 (sid, int(session)), fetch="one")
    return r[0] if r else None


def _pg_put(sid, session, topic, state, summary):
    _pg_exec("""
        INSERT INTO pg_session (student_id, session, topic, state, summary, updated)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (student_id, session) DO UPDATE
          SET topic=EXCLUDED.topic, state=EXCLUDED.state,
              summary=EXCLUDED.summary, updated=EXCLUDED.updated
    """, (sid, int(session), topic, json.dumps(state, ensure_ascii=False),
          json.dumps(summary, ensure_ascii=False), time.time()))


def _pg_list(only_session=None):
    if only_session is None:
        rows = _pg_exec("SELECT summary FROM pg_session", fetch="all")
    else:
        rows = _pg_exec("SELECT summary FROM pg_session WHERE session=%s",
                        (int(only_session),), fetch="all")
    return [r[0] for r in (rows or []) if r[0]]


def _pg_clear(session=None):
    if session is None:
        _pg_exec("DELETE FROM pg_session")
    else:
        _pg_exec("DELETE FROM pg_session WHERE session=%s", (int(session),))


# ---------------------------------------------------------------------------
# 공통 진입점
# ---------------------------------------------------------------------------
def backend() -> str:
    return "postgres" if _USE_PG else "memory"


def init():
    if _USE_PG:
        try:
            _pg_init()
            print("[store] Postgres 백엔드 준비 완료")
            return
        except Exception as e:
            print("[store] Postgres 초기화 실패 → 메모리 폴백:", e)
            globals()["_USE_PG"] = False
    print("[store] 메모리 백엔드 사용(재시작 시 상태 사라짐)")


def get_state(sid, session):
    try:
        return _pg_get(sid, session) if _USE_PG else _mem_get(sid, session)
    except Exception as e:
        print("[store] get 실패:", e); return None


def put_state(sid, session, topic, state, summary):
    try:
        (_pg_put if _USE_PG else _mem_put)(sid, session, topic, state, summary)
    except Exception as e:
        print("[store] put 실패:", e)


def list_summaries(only_session=None):
    try:
        return _pg_list(only_session) if _USE_PG else _mem_list(only_session)
    except Exception as e:
        print("[store] list 실패:", e); return []


def clear(session=None):
    try:
        (_pg_clear if _USE_PG else _mem_clear)(session)
    except Exception as e:
        print("[store] clear 실패:", e)
