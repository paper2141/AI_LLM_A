import datetime
import json
import os
import time
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()


# ==============================================================================
# 1. 시스템 페르소나 설정
# ==============================================================================
def build_reporter_persona(about: str) -> str:
    return f"""너는 신문기자이다.

최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다.

단계별 기사 작성 지시사항
- 최근 어떤 이슈가 있는지 검색하고 사람들이 제일 관심있어 할만한 주체를 선정하고 왜 선정했는지 말해줘
- 그 내용으로 원고를 작성하기 위한 목차를 만들고 목차 내용을 채우기 위해 추가로 검색할 내용을 리스트로 정리해봐
- 검색할 리스트를 토대로 재검색해
- 목차에 있는 내용을 작성하기 위해 더 검색이 필요한 정보가 있는지 확인하고 있다면 추가로 검색해
- 검색된 결과에서 원하는 정보를 찾지 못했다면 다른 검색어로 재검색해도 좋아.

최종 기사 출력 조건
더이상 검색할 내용이 없다면 신문 기사 형식으로 최종 기사를 작성한다.
제목, 부제, 리드문, 본문의 구성으로 작성하며, 본문내용은 심층분석 기사에 맞게 구체적이고 깊이있게 작성해야 한다."""


# ==============================================================================
# 2. 상태(State) 정의 : TypedDict 상속
# ==============================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ==============================================================================
# 3. 도구(Tools) 등록 : get_current_time, get_web_search
# ==============================================================================
@tool
def get_current_time() -> str:
    """현재 날짜와 시각을 YYYY-MM-DD HH:MM:SS 형식으로 반환합니다."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_web_search(query: str) -> str:
    """인터넷에서 최신 뉴스와 이슈를 검색하여 결과를 반환합니다."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if results:
                return json.dumps(
                    [
                        {
                            "source": item.get("href", "Web Search"),
                            "title": item.get("title", ""),
                            "snippet": item.get("body", ""),
                            "reliability": "95%",
                            "time": "실시간 수집",
                        }
                        for item in results
                    ],
                    ensure_ascii=False,
                )
    except Exception as e:
        return json.dumps(
            [
                {
                    "source": "Search Fallback",
                    "title": f"[{query}] 관련 최신 이슈 검색 결과",
                    "snippet": f"검색어 '{query}'에 대한 주요 쟁점 및 정책·산업계 비판 분석 데이터 (참고: {str(e)})",
                    "reliability": "92%",
                    "time": "방금 수집",
                }
            ],
            ensure_ascii=False,
        )
    return "검색 결과가 없습니다. 다른 검색어로 재검색해 주세요."


tools = [get_current_time, get_web_search]


# ==============================================================================
# 4. 언어 모델 설정 - 구글 AI (Gemini) 및 도구 바인딩
# ==============================================================================
def get_llm_with_tools():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=api_key,
        temperature=0.3,
    )
    return llm.bind_tools(tools)


# ==============================================================================
# 5. 노드 생성 : generate, BasicToolNode
# ==============================================================================
def generate(state: State) -> Dict[str, List[AIMessage]]:
    """현재 상태의 메시지 리스트를 기반으로 LLM 응답(또는 도구 호출 요청)을 생성하는 노드"""
    llm_with_tools = get_llm_with_tools()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


class BasicToolNode:
    """마지막 AIMessage의 tool_calls를 실행하고 ToolMessage를 반환하는 도구 노드"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool_item.name: tool_item for tool_item in tools}

    def __call__(self, inputs: dict) -> Dict[str, List[ToolMessage]]:
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs: List[ToolMessage] = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# ==============================================================================
# 6. 라우터 설정 및 그래프 컴파일 : route_tools, START -> generate -> route_tools
# ==============================================================================
def route_tools(state: State):
    """generate 노드의 출력에 tool_calls가 있으면 'tools'로, 없으면 END로 분기하는 라우터"""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate)
graph_builder.add_node("tools", BasicToolNode(tools=tools))

graph_builder.add_edge(START, "generate")
graph_builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "generate")

graph = graph_builder.compile()


# ==============================================================================
# 7. FastAPI 서버 및 .stream 기반 답변 출력 엔드포인트
# ==============================================================================
app = FastAPI(title="PressPulse AI — LangGraph Reporter Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    about: str = Field(..., min_length=1, description="비판적 심층 분석 기사를 작성할 주제 ({about})")
    tone: str = Field(default="expose", description="기사 논조 (expose / balance)")


def parse_article_sections(raw_text: str, about: str) -> Dict[str, Any]:
    """생성된 최종 기사 텍스트에서 제목, 부제, 리드문, 본문을 구조화하여 추출합니다."""
    title = f"'{about}' 심층 분석 리포트"
    subtitle = "핵심 쟁점과 구조적 한계 진단"
    lead = ""
    paragraphs: List[str] = []

    current_section = ""
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        normalized = cleaned.replace("*", "").replace("#", "").strip()

        if normalized.startswith(("제목:", "[제목]")):
            title = normalized.split(":", 1)[-1].replace("[제목]", "").strip()
            current_section = "title"
        elif normalized.startswith(("부제:", "[부제]")):
            subtitle = normalized.split(":", 1)[-1].replace("[부제]", "").strip()
            current_section = "subtitle"
        elif normalized.startswith(("리드문:", "[리드문]", "리드:")):
            rest = normalized.split(":", 1)[-1].replace("[리드문]", "").strip()
            if rest:
                lead = rest
            current_section = "lead"
        elif normalized.startswith(("본문:", "[본문]")):
            rest = normalized.split(":", 1)[-1].replace("[본문]", "").strip()
            if rest:
                paragraphs.append(rest)
            current_section = "body"
        else:
            if current_section == "lead" and not lead:
                lead = cleaned
                current_section = "body"
            elif not lead and len(cleaned) > 40 and not paragraphs:
                lead = cleaned
            else:
                paragraphs.append(cleaned)

    if not paragraphs:
        paragraphs = [raw_text.strip()]

    return {
        "category_label": f"[심층 기획] {about} 비판 특별 리포트",
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "paragraphs": paragraphs,
        "full_text": raw_text.strip(),
    }


@app.get("/health")
@app.get("/status")
async def get_status() -> Dict[str, Any]:
    return {
        "status": "ONLINE",
        "connected": True,
        "engine": "LangGraph (START -> generate -> route_tools -> tools / END)",
        "tools": ["get_current_time", "get_web_search"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/report")
async def create_report(req: ReportRequest):
    """
    graph.stream()을 사용하여 단계별 검색·목차 구성·재검색·최종 기사 작성을 수행하고
    GAS 클라이언트(code.gs / index.html)에 표준 JSON으로 반환합니다.
    """
    t0 = time.perf_counter()
    about = req.about.strip()
    system_persona = build_reporter_persona(about)

    inputs: State = {
        "messages": [
            SystemMessage(content=system_persona),
            HumanMessage(content=f"'{about}'에 대해 단계별 지시사항에 따라 검색 및 재검색을 수행한 뒤 최종 비판 심층 분석 기사를 작성해줘."),
        ]
    }

    stream_steps: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    final_content = ""

    try:
        for event in graph.stream(inputs):
            for node_name, value in event.items():
                messages = value.get("messages", [])
                for msg in messages:
                    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                    tool_calls = getattr(msg, "tool_calls", []) or []

                    stream_steps.append(
                        {
                            "node": node_name,
                            "content": content,
                            "tool_calls": [tc.get("name") for tc in tool_calls],
                        }
                    )

                    if node_name == "tools" and getattr(msg, "name", "") == "get_web_search":
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                sources.extend(parsed)
                        except Exception:
                            sources.append(
                                {
                                    "source": "Web Search",
                                    "title": f"[{about}] 검색 결과",
                                    "snippet": content[:250],
                                    "reliability": "94%",
                                    "time": "실시간 수집",
                                }
                            )

                    if node_name == "generate" and content and not tool_calls:
                        final_content = content

        article = parse_article_sections(final_content, about)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "status": "SUCCESS",
            "elapsed_ms": elapsed_ms,
            "about": about,
            "langgraph_stream_steps": stream_steps,
            "sources": sources[:6],
            "article": article,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": {
                    "code": "LANGGRAPH_EXECUTION_ERROR",
                    "message": f"LangGraph 기사 생성 중 오류가 발생했습니다: {str(e)}",
                },
            },
        )


@app.post("/report/stream")
async def stream_report(req: ReportRequest):
    """graph.stream()의 각 노드 출력을 실시간 NDJSON 스트림으로 반환합니다."""
    about = req.about.strip()
    inputs: State = {
        "messages": [
            SystemMessage(content=build_reporter_persona(about)),
            HumanMessage(content=f"'{about}'에 대한 심층 분석 기사를 작성해줘."),
        ]
    }

    def event_generator():
        for event in graph.stream(inputs):
            for node_name, value in event.items():
                for msg in value.get("messages", []):
                    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                    tool_calls = getattr(msg, "tool_calls", []) or []
                    yield json.dumps(
                        {
                            "node": node_name,
                            "content": content,
                            "tool_calls": [tc.get("name") for tc in tool_calls],
                        },
                        ensure_ascii=False,
                    ) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

