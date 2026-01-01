import asyncio
import sys
import os
from uuid import uuid4

# --- 路径黑魔法：确保能导入你的模块 ---
sys.path.append(os.getcwd())

from loguru import logger
# 根据你的目录结构调整导入路径
from src.knowledge.rag_service import RAGService, RAGResult
from src.models.knowledge import KnowledgeNode, OperationGuide, KnowledgeGraph, NodeType
from src.services.embedding_service import EmbeddingService

async def test_rag_flow():
    logger.info("🚀 开始测试 RAG + BGE-M3 集成流程...")

    # 1. 初始化服务
    # ------------------------------------------------------
    embedding_service = EmbeddingService()
    rag_service = RAGService()
    knowledge_graph = KnowledgeGraph()

    try:
        # 启动 Embedding 服务 (建立 HTTP Client)
        await embedding_service.initialize()
        
        # 启动 RAG 服务 (注入依赖)
        await rag_service.initialize(embedding_service, knowledge_graph)
        print("\n✅ 服务初始化成功")

        # 2. 准备模拟数据 (造一条关于“微信”的知识)
        # ------------------------------------------------------
        print("\n📝 正在索引测试数据...")
        
        # 创建一个模拟指南
        test_guide = OperationGuide(
            id=uuid4(),
            title="如何调整微信字体大小",
            app_name="微信",
            feature_name="字体设置",
            steps=[
                "打开微信，点击右下角的'我'",
                "点击'设置'选项",
                "选择'通用'",
                "点击'字体大小'",
                "拖动底部的滑块来调整字体"
            ],
            # description="帮助老年人看不清字的时候调大微信字体"
        )

        # 索引它 (这里会调用 BGE-M3 生成文档向量)
        await rag_service.index_guide(test_guide)
        print(f"✅ 数据索引完成: {test_guide.title}")

        # 3. 模拟提问 (测试检索能力)
        # ------------------------------------------------------
        query = "微信字太小了看不清怎么办"  # 注意：故意不完全匹配标题，测试语义理解
        print(f"\n❓ 正在提问: {query}")

        # 调用检索 (这里会调用 BGE-M3 生成问题向量)
        result = await rag_service.retrieve(query, top_k=1)

        # 4. 验证结果
        # ------------------------------------------------------
        print("\n📊 检索结果:")
        print(f"   - 置信度 (Confidence): {result.confidence:.4f}")
        
        if result.guides:
            top_guide = result.guides[0]
            print(f"   - 匹配到的指南: 【{top_guide.title}】")
            print("   - 生成的上下文预览:")
            print("-" * 30)
            print(result.context)
            print("-" * 30)
            
            # 简单断言
            if top_guide.id == test_guide.id:
                print("\n🎉 测试通过！成功通过 BGE-M3 语义匹配找到了正确文档。")
            else:
                print("\n❌ 测试失败：匹配到了错误的文档。")
        else:
            print("\n❌ 测试失败：没有检索到任何结果 (可能是阈值太高或 Embedding 失败)。")

    except Exception as e:
        logger.exception(f"❌ 测试过程中发生错误: {e}")
    
    finally:
        # 清理资源
        await embedding_service.close()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_rag_flow())