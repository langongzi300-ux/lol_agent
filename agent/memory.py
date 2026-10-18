"""
对话记忆模块：LLM摘要 + 最近N轮原文 的混合记忆方案

- recent：最近N轮对话原文，保证时效性和细节（如指代"他/刚才那个英雄"）
- summary：更早的对话由LLM滚动合并成摘要，防止历史无限膨胀
"""
from langchain_core.language_models import BaseChatModel


class ConversationMemory:
    def __init__(self, chat_model: BaseChatModel, max_recent_turns: int = 4, summarize_batch_turns: int = 2):
        """
        :param chat_model: 用于生成摘要的对话模型
        :param max_recent_turns: 保留原文的最近对话轮数（一轮=用户+助手各一条）
        :param summarize_batch_turns: 每攒够多少轮就触发一次摘要合并
        """
        self.chat_model = chat_model
        self.max_recent_turns = max_recent_turns
        self.summarize_batch_turns = summarize_batch_turns

        self.summary: str = ""                              # 长期摘要
        self.recent: list[dict] = []                        # 最近N轮原文 [{"role": ..., "content": ...}]
        self._pending: list[dict] = []                      # 等待攒批摘要的对话

    def get_context(self) -> tuple[str, list[dict]]:
        """返回(长期摘要, 最近N轮原文)，供组装发给模型的消息列表"""
        return self.summary, list(self.recent)

    def add_turn(self, user_query: str, assistant_answer: str) -> None:
        """对话结束后记录一轮，超出窗口的轮次自动滚动进摘要"""
        self.recent.append({"role": "user", "content": user_query})
        self.recent.append({"role": "assistant", "content": assistant_answer})

        # 超出窗口的轮次移入待摘要缓冲
        while len(self.recent) > self.max_recent_turns * 2:
            self._pending.extend(self.recent[:2])
            self.recent = self.recent[2:]

            # 攒够一批就合并进长期摘要
            while len(self._pending) >= self.summarize_batch_turns * 2:
                batch = self._pending[:self.summarize_batch_turns * 2]
                self._pending = self._pending[self.summarize_batch_turns * 2:]
                self.summary = self._merge_summary(self.summary, batch)

    def _merge_summary(self, existing_summary: str, turns: list[dict]) -> str:
        """把一批对话合并进已有摘要"""
        conversation = "\n".join(
            f"{'用户' if t['role'] == 'user' else '助手'}：{t['content']}" for t in turns
        )

        prompt = f"""请把「新发生的对话」合并进「已有对话摘要」，生成更新后的摘要。

要求：
1. 保留关键事实：用户的姓名/称呼、偏好、明确结论、讨论过的英雄及其要点；
2. 去除寒暄、重复和次要细节，总长度控制在200字以内；
3. 若已有摘要为空，则直接根据新对话生成摘要；
4. 仅输出摘要文本本身，不要输出任何解释。

已有对话摘要：
{existing_summary or '（空）'}

新发生的对话：
{conversation}

更新后的摘要："""

        return self.chat_model.invoke(prompt).content.strip()
