from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, fetch_external_data,
                                     get_lol_version_info, save_document)


class ReactAgent:
    def __init__(self):
        tools = [
            rag_summarize, fetch_external_data, get_lol_version_info, save_document,
        ]

        # 使用 LangGraph 官方预置的 ReAct Agent 构建器
        # state_modifier 会在每次调用模型前把系统提示词注入到消息列表中
        self.agent = create_react_agent(
            model=chat_model,
            tools=tools,
            state_modifier=load_system_prompts(),
        )

    def execute_stream(self, query: str):
        input_messages = {"messages": [HumanMessage(content=query)]}

        # stream_mode="values" 返回每次状态更新后的完整消息列表
        for chunk in self.agent.stream(input_messages, stream_mode="values"):
            messages = chunk.get("messages", [])
            if not messages:
                continue

            latest_message = messages[-1]
            content = getattr(latest_message, "content", None)
            if content:
                yield content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我介绍一下暗裔剑魔亚托克斯"):
        print(chunk, end="", flush=True)
