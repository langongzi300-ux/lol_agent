import os
import re
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

_rag_service: Optional[RagSummarizeService] = None


def _get_rag_service() -> RagSummarizeService:
    """延迟初始化 RAG 服务，避免模块导入时就需要 DASHSCOPE_API_KEY。"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


external_data = {}


@tool
def rag_summarize(query: str) -> str:
    """从向量存储中检索英雄联盟相关的参考资料（英雄介绍、技能、昵称等），入参query为核心检索词，以字符串形式返回检索并总结后的资料内容"""
    return _get_rag_service().rag_summarize(query)


@tool
def get_lol_version_info() -> str:
    """获取当前英雄联盟数据版本信息（如16.18.1），无入参，以纯字符串形式返回"""
    data_dir = get_abs_path("data")

    # 从数据文件名（如 英雄联盟英雄总结_v16.18.1.txt）中解析版本号
    for filename in os.listdir(data_dir):
        match = re.search(r"_v(\d+\.\d+\.\d+)", filename)
        if match:
            return f"当前英雄联盟数据版本为：{match.group(1)}"

    logger.warning("[get_lol_version_info]未能从数据文件中解析出版本号，返回默认版本")
    return "当前英雄联盟数据版本为：16.18.1"


@tool
def save_document(title: str, content: str) -> str:
    """将整理好的文档内容保存为Markdown文件，入参title为文档标题，content为整理好的正文内容，返回保存的文件路径；当用户要求记录、总结、保存对话或生成文档时调用"""
    save_dir = get_abs_path("docs")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title.strip())[:50] or "未命名文档"
    filename = f"{safe_title}_{timestamp}.md"
    filepath = os.path.join(save_dir, filename)

    document = f"# {title.strip()}\n\n> 保存时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content.strip()}\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(document)

    logger.info(f"[save_document]文档已保存：{filepath}")
    return f"文档保存成功，文件路径：{filepath}"


def generate_external_data():
    """
    加载英雄联盟全球总决赛（S赛）历史记录，按届数建立索引：
    {
        "S1": {"决赛时间": xxx, "举办地": xxx, "冠军": xxx, "决赛对阵": xxx, "冠军决胜局使用英雄": xxx, "备注": xxx},
        "S2": {...},
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.read().splitlines()[1:]:
                if not line.strip():
                    continue

                # 数据文件为制表符分隔：届数、决赛时间、举办地、冠军、决赛对阵、冠军决胜局使用英雄、备注
                arr: list[str] = line.split("\t")

                if len(arr) < 7:
                    logger.warning(f"[加载外部数据]跳过格式异常的行：{line}")
                    continue

                season: str = arr[0].strip()
                external_data[season] = {
                    "决赛时间": arr[1].strip(),
                    "举办地": arr[2].strip(),
                    "冠军": arr[3].strip(),
                    "决赛对阵": arr[4].strip(),
                    "冠军决胜局使用英雄": arr[5].strip(),
                    "备注": arr[6].strip(),
                }


def _format_season_record(season: str, record: dict) -> str:
    return (f"【{season}】决赛时间：{record['决赛时间']} | 举办地：{record['举办地']} | "
            f"冠军：{record['冠军']} | 决赛对阵：{record['决赛对阵']} | "
            f"冠军决胜局使用英雄：{record['冠军决胜局使用英雄']} | 备注：{record['备注']}")


@tool
def fetch_external_data(season: str) -> str:
    """查询英雄联盟全球总决赛（S赛）历史记录。入参season为届数（如"S1"、"S12"），传入"全部"返回所有届数的记录，以纯字符串形式返回，如果未检索到返回空字符串"""
    generate_external_data()

    season = season.strip().upper()
    # 兼容纯数字入参，如"1"视为"S1"
    if not season.startswith("S") and season.isdigit():
        season = "S" + season

    if season == "全部":
        return "\n".join(_format_season_record(s, r) for s, r in external_data.items())

    try:
        return _format_season_record(season, external_data[season])
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到届数：{season}的S赛记录")
        return ""
