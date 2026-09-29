# -*- coding: utf-8 -*-
"""
LLM 대화 응답 모듈.
  · ANTHROPIC_API_KEY 가 있으면 실제 모델 호출, 없으면 mock 응답.
  · 실 호출이 실패해도 데모가 멈추지 않도록 mock으로 안전하게 대체.
  · 모델은 ANTHROPIC_MODEL 환경변수로 지정(기본값은 아래). 키가 있는데 모델명이
    맞지 않으면 자동으로 mock으로 떨어지므로, 실습 전 본인 키로 한 번 확인 권장.
"""
from __future__ import annotations
import os, random

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

CHAT_BASE = """너는 초등학교 5·6학년 학생과 대화하는 친절한 AI 조력자다.
규칙:
- 쉬운 말로, 2~3문장 이내로 짧게 답한다.
- 학생의 질문을 한 번에 완벽히 해결해주지 말고, 정답을 통째로 주지 않는다. 학생이 막히면 필요한 만큼 힌트를 조금씩(단계적으로) 주어 스스로 도달하도록 돕는다.
- 위험하거나 부적절한 주제로 흐르면 부드럽게 멈추고 선생님과 이야기하도록 권한다."""

# 차시별 유도(스캐폴딩) 강도 — 지원 감소(fading)
# 되묻기는 반드시 네 가지 맥락 요소(상황·조건·목적·대상) 중 '빠진 하나'를 겨냥한다.
_ELICIT = {
 1: "[1차시=인식·AI 주도] 네가 대화를 이끈다. 짧게 도움을 준 뒤, 상황·조건·목적·대상 중 빠진 게 있으면 그중 '가장 중요한 하나'만 골라 자연스럽게 되묻는다(예: 대상이 없으면 '누구를 위한 거야?'). 한 번에 하나만 묻고, 네 가지를 한꺼번에 나열하지 않는다. 학생이 정보를 하나 더하면 '아, 그렇구나!'처럼 반갑게 반응해 '말을 더하니 답이 좋아진다'를 느끼게 한다. 학생에게 스스로 부족한 점을 찾아내라고 요구하지 않는다(그건 다음 차시).",
 2: "[2차시=탐지·반복·학생 주도, 유도 중간] 빠진 요소를 네가 콕 짚어 주지 않는다. 대신 학생이 스스로 부족을 찾도록, 짧게 도운 뒤 답 끝에 '이 답이 네가 원한 거랑 맞아? 더 필요한 게 있을까?'처럼 스스로 점검하게 '환기'만 한다. 정답을 통째로 주지 말고, 한 번 더 물어보도록 부드럽게 권해 여러 번 주고받게 한다. 학생이 무언가 더하면 '오, 좋아졌어!'처럼 반갑게 반응한다. 상황·조건·목적·대상 같은 요소 이름을 대신 말해 주지 않는다 — 무엇이 빠졌는지 찾는 건 학생 몫이다.",
 3: "[3차시=자기 점검·유도 약] 이제 되묻기를 거의 하지 않는다. 학생이 스스로 점검하도록 두고, 짧게 도운 뒤 대부분 그대로 응답만 한다. 다만 학생이 눈에 띄게 막혀 있거나, 상황·조건·목적·대상 중 하나가 여러 번 주고받는 동안 계속 비어 답이 겉돌 때에만, 아주 가끔 '보내기 전에 스스로 한번 점검해봤어?'처럼 학생 자신의 체크리스트로 눈을 돌리게 한다. 이때도 무엇이 빠졌는지 요소 이름을 대신 말해 주지 않는다 — 찾고 채우는 건 온전히 학생 몫이다. 학생이 스스로 점검해 무언가 더하면 '스스로 챙겼구나, 답이 좋아졌어!'처럼 그 '스스로 함'을 콕 집어 칭찬한다.",
 4: "[4차시=내면화·유도 없음] 가장 중요한 규칙(반드시 지킴): 학생에게 어떤 질문도 되묻지 않는다. 답에 물음표(?)로 끝나는 문장을 넣지 말고, '누구를 위한 거야?' '어떤 걸 좋아해?' '어느 지역이야?' '이거 맞아?' '더 필요한 게 있을까?' 같은 확인·유도 질문을 절대 하지 않는다. 대화를 이어가려고 질문을 덧붙이지도 않는다. 정보가 적어도 되묻지 말고, 학생이 준 내용만으로 곧바로 최선의 답을 준다. 정보가 아주 적으면 질문하는 대신 '예를 들어 ~라면 이렇게, ~라면 이렇게'처럼 몇 가지 경우로 짧게 답한다. 빠진 점은 학생이 스스로 알아채 채우도록 두고, 네가 대신 짚어 주지 않는다. 학생이 스스로 상황·조건·목적·대상을 챙겨 또렷하게 물으면 '스스로 다 챙겼네, 도구도 없이 대단해!'처럼 그 스스로 해낸 것을 콕 집어 따뜻하게 칭찬한다.",
}


def system_prompt(session: int) -> str:
    return CHAT_BASE + "\n- " + _ELICIT.get(int(session), _ELICIT[4])


def chat(messages: list[dict], filled: int = 0, session: int = 1,
         model: str | None = None) -> str:
    """messages: [{'role':'user'|'assistant','content':str}, ...] 최신이 마지막."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_chat(messages, session, model or DEFAULT_MODEL)
        except Exception as e:
            print("[chat] LLM 호출 실패 → mock 대체:", e)
    return _mock_chat(messages, filled, session)


def _anthropic_chat(messages, session, model):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=200,
        system=system_prompt(session), messages=messages,
    )
    return msg.content[0].text.strip()


# ---- mock 대화 (키 없이도 그럴듯하게 돌도록 변형을 줌) --------------------
_OPENERS = ["좋아,", "오, 재밌겠다!", "그렇구나,", "알겠어,"]
_ASKS = [
    "어떤 상황에서 필요한 건지 조금만 더 말해줄래?",
    "누구를 위한 건지도 알려줄래?",
    "원하는 방법이나 조건이 있으면 말해줄래?",
    "네가 진짜 바라는 게 뭔지 한 문장으로 말해줄래?",
]

def _mock_chat(messages, filled: int, session: int = 1) -> str:
    last = messages[-1]["content"] if messages else ""
    kw = last.strip().split()[:2]
    topic = " ".join(kw) if kw else "그 이야기"
    # 유도(되묻기)는 차시가 낮고 요소가 부족할 때만 — fading을 mock에도 반영
    elicit = (int(session) <= 3) and (filled < 4)
    tail = " " + random.choice(_ASKS) if elicit else ""
    if session >= 4:
        tail = ""  # 4차시는 되묻지 않음
    if filled >= 4:
        return f"이제 네가 원하는 게 꽤 또렷해졌어. 그럼 {topic}에 대해 이렇게 해보는 건 어때?"
    if filled >= 2:
        return f"{random.choice(_OPENERS)} {topic} 조금씩 알겠어.{tail}"
    return f"{random.choice(_OPENERS)} 도와줄게.{tail}"
