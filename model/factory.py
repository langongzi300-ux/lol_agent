from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from utils.config_handler import rag_conf

# 自动加载项目根目录下的 .env 文件（如果存在）
load_dotenv()

# DashScope text-embedding-v4 不在 langchain_community 0.3.7 的 BATCH_SIZE 映射里，
# 默认 batch size 25 会触发 API 400: batch size should not be larger than 10。
# 手动补丁为 10，让 embedding 分批调用。
import langchain_community.embeddings.dashscope as _dashscope_embeddings_module
_dashscope_embeddings_module.BATCH_SIZE["text-embedding-v4"] = 10


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class _LazyModel:
    """
    延迟初始化包装器。
    解决模型在模块导入时就立即实例化、导致缺少 API Key 时连 import 都失败的问题。
    真正的 ChatTongyi / DashScopeEmbeddings 实例会在第一次被使用时才创建。
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._instance: Optional[Any] = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        return f"LazyModel({self._get_instance()!r})"


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 使用延迟包装，避免导入时就创建实例
chat_model = _LazyModel(ChatModelFactory().generator)

embed_model = _LazyModel(EmbeddingsFactory().generator)


def get_chat_model() -> Optional[BaseChatModel]:
    """直接返回真正的 ChatTongyi 实例（可用于 LangChain 的 `|` 管道）。"""
    return ChatModelFactory().generator()


def get_embed_model() -> Optional[Embeddings]:
    """直接返回真正的 DashScopeEmbeddings 实例。"""
    return EmbeddingsFactory().generator()
