# -*- coding: utf-8 -*-
"""
LLM 대화 응답 모듈.
  · ANTHROPIC_API_KEY 가 있으면 실제 모델 호출, 없으면 mock 응답.
  · 실 호출이 실패해도 데모가 멈추지 않도록 mock으로 안전하게 대체.
  · 모델은 ANTHROPIC_MODEL 환경변수로 지정(기본값은 아래). 키가 있는데 모델명이
    맞지 않으면 자동으로 mock으로 떨어지므로, 실습 전 본인 키로 한 번 확인 권장.
"""
from __future__ import annotations
import os, random, time

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
# 답변 최대 토큰 — 너무 낮으면 한국어 답이 중간에 잘린다(기존 200 → 512).
# 상한일 뿐이라 짧은 답은 알아서 짧게 끝나므로 비용은 거의 그대로.
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", 512))

# 실 LLM 호출이 '예외로' mock 대체된 마지막 시각(키가 아예 없는 정상 mock 모드는 제외).
# 관리자/교사 화면에서 '조용한 품질 저하'를 감지하는 데 쓴다.
_DEGRADED_AT = 0.0


def degraded_recently(window: int = 120) -> bool:
    """최근 window초 안에 실 LLM 호출이 실패해 mock으로 떨어진 적이 있으면 True."""
    return _DEGRADED_AT > 0 and (time.time() - _DEGRADED_AT) < window

CHAT_BASE = """너는 초등학교 5·6학년 학생과 대화하는 친절한 AI 조력자다.
규칙:
- 초등학생에게 말하듯 친근한 반말로, 쉬운 말로, 2~3문장 이내로 짧게 답한다. 존댓말('~요','~습니다')을 쓰지 않고 모든 차시에서 반말 톤을 일관되게 유지한다.
- 학생이 준 정보만으로도 실제로 도움이 되는 답을 준다. 정답을 일부러 감추거나 힌트만 찔끔 주는 식으로 굴지 않는다. (답을 얼마나 주고 학생에게 되물을지 말지는 오직 아래 '차시별 규칙'만 따른다.)
- 위험하거나 부적절한 주제로 흐르면 부드럽게 멈추고 선생님과 이야기하도록 권한다."""

# 차시별 유도(스캐폴딩) 강도 — 지원 감소(fading)
# 되묻기는 반드시 네 가지 맥락 요소(상황·조건·목적·대상) 중 '빠진 하나'를 겨냥한다.
_ELICIT = {
 1: "[1차시=인식·AI 주도] 네가 대화를 이끈다. 짧게 도움을 준 뒤, 상황·조건·목적·대상 중 빠진 게 있으면 그중 '가장 중요한 하나'만 골라 자연스럽게 되묻는다(예: 대상이 없으면 '누구를 위한 거야?'). 한 번에 하나만 묻고, 네 가지를 한꺼번에 나열하지 않는다. 학생이 정보를 하나 더하면 '아, 그렇구나!'처럼 반갑게 반응해 '말을 더하니 답이 좋아진다'를 느끼게 한다. 학생에게 스스로 부족한 점을 찾아내라고 요구하지 않는다(그건 다음 차시).",
 2: "[2차시=탐지·반복·학생 주도, 유도 중간] 반드시 지킬 규칙: 빠진 요소를 콕 집어 묻지 않는다. '어떤 상황이야?' '누구를 위한 거야?' '어떤 조건이야?' '뭘 가르치고 싶어?'처럼 특정 요소를 지목해 캐묻는 질문을 하지 않고, 상황·조건·목적·대상 같은 요소 이름을 말하지도 않는다(그건 1차시 방식이다). 대신 반드시 이 순서로 답한다 — ① 먼저 학생 질문에 짧지만 실제로 도움이 되는 답을 준다(질문만 던지고 끝내지 않는다). ② 그런 다음 답 끝에, 아래 예시 중 하나를 골라 거의 그대로 짧게 한마디만 붙여 '환기'한다(여러 문장을 억지로 합치거나 변형해서 어색하게 만들지 말 것): '이 답이 네가 찾던 거 맞아?' / '이대로 괜찮아? 더 필요한 게 있으면 말해줘.' / '혹시 더 알려주고 싶은 게 있어?' 무엇이 빠졌는지 찾는 건 학생 몫이므로 네가 대신 짚어 주지 않는다. 학생이 스스로 무언가 더하면 '오, 좋아졌어!'처럼 반갑게 반응하고, 또 한 번 점검해 보라고 부드럽게 권해 여러 번 주고받게 한다.",
 3: "[3차시=자기 점검·유도 약] 반드시 지킬 규칙: 되묻는 질문을 하지 않는다. '어떤 상황이야?' '누구를 위한 거야?' '뭘 가르치고 싶어?'처럼 빠진 요소를 캐묻거나 요소 이름을 말하지 않는다. 먼저 학생 질문에 도움이 되는 답을 바로 주고, 대부분은 답만 하고 끝낸다(되묻지 않는다). 다만 학생이 눈에 띄게 막혀 있거나 답이 계속 뻔하게 겉돌 때에만, 아주 가끔 답 끝에 '보내기 전에 스스로 한번 점검해봤어?' 한마디만 덧붙여 학생 자신의 머릿속 체크리스트로 눈을 돌리게 한다(이때도 무엇이 빠졌는지는 말해 주지 않는다). 2차시와 달리 매번 환기하지 않고, 대개는 조용히 답만 준다. 학생이 스스로 점검해 무언가 더하면 '스스로 챙겼구나, 답이 좋아졌어!'처럼 그 스스로 함을 콕 집어 칭찬한다.",
 4: "[4차시=내면화·유도 없음] 가장 중요한 규칙(반드시 지킴): 학생에게 어떤 질문도 되묻지 않는다. 답에 물음표(?)로 끝나는 문장을 넣지 말고, '누구를 위한 거야?' '어떤 걸 좋아해?' '어느 지역이야?' '이거 맞아?' '더 필요한 게 있을까?' 같은 확인·유도 질문을 절대 하지 않는다. 대화를 이어가려고 질문을 덧붙이지도 않는다. 정보가 적어도 되묻지 말고, 학생이 준 내용만으로 곧바로 최선의 답을 준다. 정보가 아주 적으면 질문하는 대신 '예를 들어 ~라면 이렇게, ~라면 이렇게'처럼 몇 가지 경우로 짧게 답한다. 빠진 점은 학생이 스스로 알아채 채우도록 두고, 네가 대신 짚어 주지 않는다. 학생이 스스로 상황·조건·목적·대상을 챙겨 또렷하게 물으면 '스스로 다 챙겼네, 도구도 없이 대단해!'처럼 그 스스로 해낸 것을 콕 집어 따뜻하게 칭찬한다.",
}


# ---- 본실험(2026-12) 방식: 2~4차시는 '답만' 만들고, 점검 한마디는 시스템이 붙인다 ----------
# 예비실험에서는 2차시 환기·3차시 '막혔을 때만' 점검을 모델이 스스로 판단해 넣었다.
# 그러면 조건(점검 요청 켬/끔)의 차이가 모델의 그때그때 판단에 섞이고, 4차시에도 질문형 응답이 남았다.
# 본실험에서는 2~4차시 모두 아래 '답만' 규칙으로 답을 만들고,
#   · 2차시 점검 요청은 조건이 켜진 학생에게만 시스템이 정해진 문장 하나를 덧붙이고,
#   · 3차시 점검 한마디는 시스템이 '막힘'을 판정했을 때만 덧붙이며,
#   · 답에 질문이 섞이면 시스템이 다시 생성한다(app.py의 _generate_reply).
NO_QUESTION_RULE = (
    "가장 중요한 규칙(반드시 지킴): 학생에게 어떤 질문도 하지 않는다. 물음표(?)로 끝나는 문장, "
    "'~해줄래', '~말해줘', '~알려줘'처럼 학생에게 정보를 더 청하는 말, '이거 맞아?', '더 필요한 게 있을까?' 같은 "
    "확인·유도 질문을 넣지 않는다. 대화를 이어가려고 질문을 덧붙이지도 않는다. 상황·조건·목적·대상 같은 요소 이름을 "
    "말하거나 무엇이 빠졌는지 짚어 주지 않는다. 정보가 적어도 되묻지 말고, 학생이 준 내용만으로 곧바로 최선의 답을 준다. "
    "정보가 아주 적으면 질문하는 대신 '예를 들어 ~라면 이렇게, ~라면 이렇게'처럼 몇 가지 경우로 짧게 답한다.")

_ANSWER_ONLY = {
 2: "[2차시] " + NO_QUESTION_RULE + " 학생 질문에 짧지만 실제로 도움이 되는 답을 준다. "
    "학생이 스스로 무언가 더하면 '오, 좋아졌어!'처럼 반갑게 반응한다.",
 3: "[3차시] " + NO_QUESTION_RULE + " 학생 질문에 도움이 되는 답을 바로 준다. "
    "학생이 스스로 무언가 더하면 '스스로 챙겼구나, 답이 좋아졌어!'처럼 그 스스로 함을 콕 집어 칭찬한다.",
 4: "[4차시] " + NO_QUESTION_RULE + " 학생이 스스로 상황·조건·목적·대상을 챙겨 또렷하게 물으면 "
    "'스스로 다 챙겼네, 도구도 없이 대단해!'처럼 그 스스로 해낸 것을 콕 집어 따뜻하게 칭찬한다.",
}

# 다시 생성할 때 덧붙이는 지시(직전 답에 질문이 들어갔을 때)
STRICT_RETRY = ("\n- [다시 쓰기] 방금 쓴 답에 학생에게 묻는 말이 들어갔다. 같은 내용을 질문 없이 다시 쓴다. "
                "모든 문장을 '~야', '~해', '~돼'처럼 평서문이나 권유문으로 끝낸다.")


# 계속 되묻기 조건(비교 조건): 1~4차시 내내 1차시와 같은 직접 되묻기를 한다.
_ALWAYS = (_ELICIT[1].replace("[1차시=인식·AI 주도]", "[직접 되묻기·AI 주도]")
                     .replace("(그건 다음 차시)", ""))


def answer_only(session: int, support: str = "fade") -> bool:
    """이 차시가 '답만' 방식(시스템이 질문을 막는 차시)인가. 계속 되묻기 조건은 늘 아니다."""
    return int(session) >= 2 and support != "always"


def system_prompt(session: int, strict: bool = False, support: str = "fade") -> str:
    s = int(session)
    if support == "always":
        rule = _ALWAYS
    else:
        rule = _ANSWER_ONLY.get(s, _ANSWER_ONLY[4]) if answer_only(s) else _ELICIT[1]
    return CHAT_BASE + "\n- " + rule + (STRICT_RETRY if strict else "")


def system_prompt_pilot(session: int) -> str:
    """예비실험(2026-09~10) 때 쓴 차시별 프롬프트 — 논문 부록·비교용으로 남겨 둠(운영에는 쓰지 않음)."""
    return CHAT_BASE + "\n- " + _ELICIT.get(int(session), _ELICIT[4])


def chat(messages: list[dict], filled: int = 0, session: int = 1,
         model: str | None = None, strict: bool = False, support: str = "fade") -> str:
    """messages: [{'role':'user'|'assistant','content':str}, ...] 최신이 마지막.
       strict=True: 질문이 섞여 다시 생성할 때(더 강한 지시를 붙임)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_chat(messages, session, model or DEFAULT_MODEL, strict, support)
        except Exception as e:
            global _DEGRADED_AT
            _DEGRADED_AT = time.time()      # 실 호출 실패 → 품질저하 신호 기록
            print("[chat] LLM 호출 실패 → mock 대체:", e)
    return _mock_chat(messages, filled, session, support)


def _anthropic_chat(messages, session, model, strict=False, support="fade"):
    import anthropic
    # 타임아웃·재시도 제한 → 느린 호출이 스레드를 오래 붙잡아 서버가 멈추는 것 방지
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 timeout=25.0, max_retries=1)
    msg = client.messages.create(
        model=model, max_tokens=MAX_REPLY_TOKENS,
        system=system_prompt(session, strict, support), messages=messages,
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

def _mock_chat(messages, filled: int, session: int = 1, support: str = "fade") -> str:
    last = messages[-1]["content"] if messages else ""
    kw = last.strip().split()[:2]
    topic = " ".join(kw) if kw else "그 이야기"
    # mock: 1차시만 되묻는다(2~4차시 점검 한마디는 실제 운영과 같이 시스템이 붙임).
    # MOCK_ASK_RATE>0이면 2~4차시에도 일부러 질문을 섞어 '다시 생성' 경로를 시험할 수 있다.
    elicit = (int(session) == 1 or support == "always") and (filled < 4)
    if int(session) >= 2 and random.random() < float(os.environ.get("MOCK_ASK_RATE", "0")):
        elicit = True
    tail = " " + random.choice(_ASKS) if elicit else ""
    if filled >= 4:
        return f"이제 네가 원하는 게 꽤 또렷해졌어. 그럼 {topic}은(는) 이렇게 해보면 좋아."
    if filled >= 2:
        return f"{random.choice(_OPENERS)} {topic} 조금씩 알겠어.{tail}"
    return f"{random.choice(_OPENERS)} 도와줄게.{tail}"
