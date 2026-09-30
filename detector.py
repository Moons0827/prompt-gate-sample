# -*- coding: utf-8 -*-
"""
prompt-gate 확장판 — 탐지 엔진 (Detection Engine)
====================================================

하나의 요소 판별 엔진을 두 곳에 재사용한다.
  · 실시간(수업 중): 학생 발화에서 빠진 맥락 요소를 찾아 '넛지'를 만든다.
  · 사후(분석): 대화 로그를 자동 코딩해 코딩 시트(턴별) 스키마로 떨군다.

표적 요소 4개: 상황(situation) · 조건(condition) · 목적(purpose) · 대상(target)

구성
  detect_elements() : LLM(또는 mock)로 발화의 4요소 유무를 판별 → 구조화 JSON
  NudgePolicy       : 차시별 넛지 강도(지원 감소)를 파라미터로 관리
  SafetyFilter      : 입력 단계 안전 사전 필터(위험 시 교사 에스컬레이션)
  ConversationCoder : 대화 1건을 턴 단위로 누적 코딩(계기 자발/넛지 판별 포함)

의존성: 표준 라이브러리만으로 mock 모드 동작. 실 LLM은 anthropic 또는 openai SDK.
사용:  python detector.py        # 오프라인 mock 데모 실행

주의: mock 판별기는 데모/테스트 전용이다. 연구·수업 투입 시 반드시 실 LLM +
      '탐지 엔진 검증 프로토콜'(요소별 F1, 엔진-사람 κ, 자기일관성)을 거친다.
"""

from __future__ import annotations
import os
import re
import json
import csv
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import Optional, Callable

ELEMENTS = ["상황", "조건", "목적", "대상"]

# ---------------------------------------------------------------------------
# 1) LLM 클라이언트 (실 호출부) — 키 없으면 mock으로 자동 대체
# ---------------------------------------------------------------------------

DETECT_SYSTEM_PROMPT = """너는 초등학교 5·6학년 학생이 AI에게 보낸 대화를 분석하는 교육 연구 보조자다.
학생이 '지금까지의 대화 전체'에서 다음 네 가지 맥락 요소를 제시했는지 각각 판단하라.
- 상황: 무슨 일/맥락에서인지 (예: "학교 간 사이에", "쉬는시간에")
- 조건: 제약·방식·수단 (예: "집에 있는 재료로", "편지로")
- 목적: 무엇을 원하는지 (예: "화해하고 싶어", "~하는 법")
- 대상: 누구를/누구에게 (예: "5개월 푸들", "예민한 친구")

또한 같은 호출에서 '이번 학생 발화'의 안전 위험도 함께 판정하라(비용 절감을 위해 요소 탐지와 통합).
다음 중 하나라도 실제로 해당하면 risk=true, 아니면 false:
- 자해·정서위험: 자해·자살 암시, 사라지고 싶음, 심한 정서 위기
- 폭력: 남을 해치려는 의도·협박 / 성: 성적·선정적 / 개인정보: 주소·전화·주민번호·비밀번호 노출 / 괴롭힘: 따돌림·괴롭힘
맥락으로 판단하고 애매하면 과잉 차단하지 않되, 자해·자살 신호는 조금이라도 의심되면 risk=true.

규칙:
1) 키워드가 아니라 '내용'으로 판단한다. 막연하면 없음(false)으로 본다.
2) 각 요소는 대화에 한 번이라도 실질적으로 등장했으면 true.
3) 반드시 아래 JSON 형식으로만 답한다. 설명 문장을 덧붙이지 않는다.

{"상황": bool, "조건": bool, "목적": bool, "대상": bool,
 "confidence": 0.0~1.0, "risk": bool,
 "category": "자해·정서위험|폭력|성|개인정보|괴롭힘|없음", "rationale": "한 줄 근거"}"""


def _build_user_prompt(history: list[str], utterance: str) -> str:
    hist = "\n".join(f"- {h}" for h in history) if history else "(없음)"
    return (f"[이전 학생 발화]\n{hist}\n\n"
            f"[이번 학생 발화]\n{utterance}\n\n"
            f"위 대화 전체 기준으로 네 요소의 유무를 판정하라.")


def _llm_detect_anthropic(history, utterance, model: str) -> dict:
    import anthropic  # 실 환경에서만 임포트
    # 타임아웃·재시도 제한 → 느린 호출이 스레드를 오래 붙잡아 서버가 멈추는 것 방지
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 timeout=25.0, max_retries=1)
    msg = client.messages.create(
        model=model, max_tokens=300,
        system=DETECT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(history, utterance)}],
    )
    text = msg.content[0].text
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"JSON 파싱 실패: {text[:120]}")
    return json.loads(m.group(0))


# ---- mock 판별기 (오프라인 데모/테스트 전용) --------------------------------
_MOCK_CUES = {
    "상황": ["때", "사이에", "시간", "중에", "동안", "상황", "학교", "집에서", "쉬는"],
    "조건": ["으로", "로 ", "없이", "말고", "대신", "편지", "재료", "예산", "분량", "쉽게"],
    "목적": ["하고 싶", "싶어", "알려", "방법", "어떻게", "하려면", "원해", "만들"],
    "대상": ["친구", "동생", "엄마", "선생", "강아지", "푸들", "형", "누나", "반 ", "걔"],
}


def _mock_detect(history, utterance, **_) -> dict:
    text = " ".join(history + [utterance])
    found = {e: any(c in text for c in cues) for e, cues in _MOCK_CUES.items()}
    conf = 0.55  # mock은 확신도 낮게 표시 → 사람 확인 플래그 유도
    # mock은 안전 판정을 하지 않는다(키워드 Layer1이 담당) → risk=False
    return {**found, "confidence": conf, "risk": False, "category": "없음",
            "rationale": "MOCK 규칙 기반(연구용 아님)"}


def detect_elements(history: list[str], utterance: str,
                    use_llm: Optional[bool] = None, model: Optional[str] = None,
                    samples: int = 1) -> dict:
    """발화의 4요소 유무를 판별. samples>1이면 self-consistency(다수결)로 안정화.
    실 LLM 호출이 실패하면 데모가 멈추지 않도록 mock으로 자동 대체한다."""
    if model is None:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _real():
        try:
            return _llm_detect_anthropic(history, utterance, model)
        except Exception as e:
            print("[detect] LLM 호출 실패 → mock 대체:", e)
            return _mock_detect(history, utterance)

    fn = _real if use_llm else (lambda: _mock_detect(history, utterance))

    if samples <= 1:
        return fn()
    # 자기일관성: 각 요소를 다수결, confidence는 일치율
    runs = [fn() for _ in range(samples)]
    out = {}
    for e in ELEMENTS:
        votes = [bool(r.get(e)) for r in runs]
        out[e] = Counter(votes).most_common(1)[0][0]
        out.setdefault("_agree", {})[e] = votes.count(out[e]) / len(votes)
    out["confidence"] = sum(out["_agree"].values()) / len(ELEMENTS)
    # 안전(risk)은 보수적으로: 한 번이라도 위험이면 위험으로 본다
    out["risk"] = any(bool(r.get("risk")) for r in runs)
    out["category"] = next((r.get("category") for r in runs if r.get("risk")), "없음")
    out["rationale"] = f"self-consistency n={samples}"
    return out


# ---------------------------------------------------------------------------
# 2) 넛지 정책 — 차시별 강도(지원 감소)를 파라미터로
# ---------------------------------------------------------------------------

@dataclass
class NudgePolicy:
    session: int                    # 1~4 차시
    conf_gate: float = 0.6          # 확신도 이 값 미만이면 사람 확인 플래그

    def intensity(self) -> str:
        return {1: "high", 2: "mid", 3: "low", 4: "off"}.get(self.session, "off")

    def make_nudge(self, detected: dict) -> Optional[str]:
        """빠진 요소가 있으면 강도에 맞는 '주의환기' 넛지를 반환. 정답은 주지 않는다."""
        level = self.intensity()
        if level == "off":
            return None
        missing = [e for e in ELEMENTS if not detected.get(e)]
        if not missing:
            return None
        e = missing[0]  # 한 번에 하나만
        if level == "high":     # 빠진 요소를 직접 짚음
            templates = {"상황": "어떤 상황에서 그런지도 말해줄래?",
                         "조건": "어떤 방법이나 조건이 있으면 좋을지 알려줄래?",
                         "목적": "네가 진짜로 원하는 게 뭔지 한 번 더 말해줄래?",
                         "대상": "누구를 위한 건지(또는 누구인지) 알려줄래?"}
            return templates[e]
        if level == "mid":      # 정답 대신 스스로 점검하게
            return "지금 받은 답이 네가 원한 거랑 딱 맞아? 부족한 게 있으면 한 번 더 물어보자."
        if level == "low":      # 정체·위험일 때만(호출측에서 판단), 최소 환기
            return "조금 더 자세히 말하면 더 좋은 답을 받을 수 있어."
        return None


# ---------------------------------------------------------------------------
# 3) 안전 필터 — 입력 사전 검사 (위험 시 교사 에스컬레이션, fail-closed)
# ---------------------------------------------------------------------------

_SAFETY_CATEGORIES = {
    "자해·정서위험": ["죽고", "자해", "사라지고 싶", "죽고싶", "때리고 싶"],
    "폭력": ["때릴", "죽일", "부술", "협박"],
    "성": ["성인물", "야한"],
    "개인정보": ["주민등록", "전화번호", "집주소", "비밀번호"],
    "괴롭힘": ["따돌", "왕따", "놀려서", "괴롭"],
}


@dataclass
class SafetyResult:
    ok: bool
    level: str          # "clear" | "escalate"
    category: Optional[str] = None


class SafetyFilter:
    """Layer 1(입력)·Layer 2(AI 출력) 공용 — 키워드 기반 빠른 게이트(fail-closed)."""
    def check(self, utterance: str) -> SafetyResult:
        for cat, cues in _SAFETY_CATEGORIES.items():
            if any(c in utterance for c in cues):
                return SafetyResult(ok=False, level="escalate", category=cat)
        return SafetyResult(ok=True, level="clear")


# ---- Layer 3: LLM 기반 2차 안전 분류기 (키워드가 놓친 우회·간접 표현 보완) ----
SAFETY_SYSTEM_PROMPT = """너는 초등학생과 AI의 대화 안전을 점검하는 보조자다.
학생의 발화가 다음 중 하나라도 실제로 해당하면 위험으로 본다.
- 자해·정서위험: 자해·자살 암시, 사라지고 싶음, 심한 정서 위기
- 폭력: 남을 해치려는 의도·협박
- 성: 성적/선정적 내용
- 개인정보: 이름 외 주소·전화번호·주민번호·비밀번호 등 노출
- 괴롭힘: 따돌림·괴롭힘 정황
맥락으로 판단하고, 애매하면 과잉 차단하지 말고 안전으로 본다.
단, 자해·자살 신호는 조금이라도 의심되면 위험으로 본다.
반드시 아래 JSON만 출력한다(설명 금지).
{"risk": true/false, "category": "자해·정서위험|폭력|성|개인정보|괴롭힘|없음"}"""


def _llm_safety_anthropic(text: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=60,
        system=SAFETY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return _parse_json(msg.content[0].text)


def deep_safety_llm(text: str, model: Optional[str] = None) -> Optional[SafetyResult]:
    """LLM으로 입력을 2차 점검. 위험이면 SafetyResult(ok=False), 안전이면 ok=True,
    키가 없거나 호출 실패면 None(=키워드 결과 유지)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    try:
        d = _llm_safety_anthropic(text, model)
        if d.get("risk"):
            return SafetyResult(ok=False, level="escalate", category=d.get("category") or "위험")
        return SafetyResult(ok=True, level="clear")
    except Exception as e:
        print("[safety] LLM 2차 점검 실패 → 키워드 결과 유지:", e)
        return None


# ---------------------------------------------------------------------------
# 4) 대화 코더 — 대화 1건을 턴 단위로 누적 코딩 (코딩 시트 스키마와 동일)
# ---------------------------------------------------------------------------

LOG_FIELDS = ["학생ID", "차시", "주제", "턴", "학생발화",
              "상황", "조건", "목적", "대상", "이번턴신규요소수",
              "행위유형", "개입", "계기", "비고"]


@dataclass
class TurnLog:
    학생ID: str; 차시: int; 주제: str; 턴: int; 학생발화: str
    상황: int = 0; 조건: int = 0; 목적: int = 0; 대상: int = 0
    이번턴신규요소수: int = 0
    행위유형: str = ""; 개입: str = "없음"; 계기: str = "없음"; 비고: str = ""


class ConversationCoder:
    """한 학생·한 차시의 대화를 순차 코딩. 실시간·사후 양쪽에서 동일하게 쓴다."""

    def __init__(self, 학생ID: str, 차시: int, 주제: str,
                 policy: Optional[NudgePolicy] = None, samples: int = 1,
                 use_llm: Optional[bool] = None, safety_llm: bool = False):
        self.meta = (학생ID, 차시, 주제)
        self.policy = policy or NudgePolicy(session=차시)
        self.samples = samples
        self.use_llm = use_llm
        self.safety_llm = safety_llm     # Layer 3: LLM 2차 입력 점검 사용 여부
        self.safety = SafetyFilter()
        self._history: list[str] = []
        self._cum = {e: False for e in ELEMENTS}   # 누적 요소 상태
        self._turn = 0
        self._last_nudged = False                  # 직전 턴에 넛지가 나갔는가
        self.logs: list[TurnLog] = []

    def step(self, utterance: str, elicited_prev: bool = False) -> dict:
        """학생 발화 한 턴 처리 → {log, safety} 반환.
        elicited_prev: 직전 AI 응답이 되묻기(유도)를 포함했는가.
        유도는 AI 대화 속에서 이뤄지므로 별도 넛지를 반환하지 않는다."""
        self._turn += 1
        학생ID, 차시, 주제 = self.meta

        safety = self.safety.check(utterance)             # Layer 1: 키워드(입력)
        # 요소 탐지 — 같은 호출에서 안전(risk)까지 함께 판정(턴당 LLM 호출 3→2로 절감)
        detected = detect_elements(self._history, utterance,
                                   use_llm=self.use_llm, samples=self.samples)
        # Layer 3(LLM 2차 안전): 탐지 호출에 통합됨 → 추가 호출 없이 위험이면 교사 에스컬레이션
        if safety.ok and detected.get("risk"):
            safety = SafetyResult(ok=False, level="escalate",
                                  category=detected.get("category") or "위험")

        # 이번 턴에 '처음' 참이 된 요소만 신규로 계산
        newly = [e for e in ELEMENTS if detected.get(e) and not self._cum[e]]
        for e in newly:
            self._cum[e] = True

        # 계기: 직전 AI가 되물은 뒤 요소가 늘었으면 '유도', 아니면 '자발'
        if newly:
            계기 = "유도" if elicited_prev else "자발"
        else:
            계기 = "없음"

        # 행위유형(휴리스틱)
        if self._turn == 1:
            행위 = "신규작성"
        elif newly:
            행위 = "발전적수정"
        else:
            행위 = "단순반복"

        log = TurnLog(학생ID, 차시, 주제, self._turn, utterance,
                      *[int(detected.get(e, False)) for e in ELEMENTS],
                      이번턴신규요소수=len(newly), 행위유형=행위, 계기=계기)

        # 개입 기록
        if not safety.ok:
            log.개입 = "교사되돌림"; log.비고 = f"안전:{safety.category}"
        else:
            if elicited_prev:
                log.개입 = "AI유도"
            if detected.get("confidence", 1.0) < self.policy.conf_gate:
                log.비고 = (log.비고 + " 사람확인요망").strip()

        self._history.append(utterance)
        self.logs.append(log)
        return {"log": log, "safety": safety}

    def elements(self) -> dict:
        """현재 누적된 요소 상태(4개 bool)를 반환. 프런트 요소 표시에 사용."""
        return dict(self._cum)

    # --- 대화 수준 집계 ----------------------------------------------------
    def summary(self) -> dict:
        cum = sum(1 for v in self._cum.values() if v)
        자발 = sum(l.이번턴신규요소수 for l in self.logs if l.계기 == "자발")
        유도 = sum(l.이번턴신규요소수 for l in self.logs if l.계기 == "유도")
        ratio = 자발 / (자발 + 유도) if (자발 + 유도) else 0.0
        conv_type = self._classify(cum)
        return {"요소누적도": cum, "자발요소수": 자발, "유도요소수": 유도,
                "자발비율": round(ratio, 2), "대화유형": conv_type,
                "총턴수": len(self.logs)}

    def _classify(self, cum: int) -> str:
        turns = len(self.logs)
        repeats = sum(1 for l in self.logs if l.행위유형 == "단순반복")
        if turns <= 2 and cum <= 1:
            return "단발수용형"
        if repeats >= max(2, turns // 2) and cum <= 2:
            return "반복정체형"
        if cum >= 3:
            return "누적발전형"
        return "부분발전형"

    def to_csv(self, path: str):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            for l in self.logs:
                w.writerow(asdict(l))


# ---------------------------------------------------------------------------
# 5) 오프라인 데모 (mock) — 민수 4차시 대화 자동 코딩
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 탐지 엔진 데모 (MOCK 모드 — 연구용 아님) ===\n")

    coder = ConversationCoder("민수", 차시=4, 주제="친구랑 화해하기")
    turns = [
        "친구랑 싸웠는데 화해하고 싶어",
        "쉬는시간에 내가 별명 불러서 걔가 삐졌어",
        "걔가 좀 예민해서 직접 말하면 더 화낼 것 같아, 편지로 하면 어때?",
    ]
    for u in turns:
        r = coder.step(u)
        L = r["log"]
        flags = "".join(name if getattr(L, name) else "·" for name in ELEMENTS)
        print(f"[턴{L.턴}] {u}")
        print(f"      요소[{flags}] 신규{L.이번턴신규요소수} · {L.행위유형} · 계기:{L.계기}")
        if not r["safety"].ok:
            print(f"      ⚠ 안전 에스컬레이션: {r['safety'].category}")
        print()

    s = coder.summary()
    print("--- 대화 수준 집계 ---")
    for k, v in s.items():
        print(f"  {k}: {v}")

    out = "민수_4차시_자동코딩.csv"
    coder.to_csv(out)
    print(f"\n턴별 코딩 CSV 저장: {out}  (코딩 시트 '턴별코딩' 탭에 그대로 붙습니다)")
