# lol_agent
一个英雄联盟专业智能问答助手，简单实现基于LangChain的agent项目
Python≥ 3.10
DashScope API Key（[阿里云百炼](https://bailian.console.aliyun.com/) 申请）

# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key"

# Windows (CMD)
set DASHSCOPE_API_KEY=your-api-key

# 初始化知识库
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"

# 启动app
streamlit run app.py

http://localhost:8501

首次运行需确保DashScope API Key已设置且`data/`目录下有知识库文档
