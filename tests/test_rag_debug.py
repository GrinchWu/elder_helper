"""RAG服务调试测试脚本 - 诊断为什么RAG搜索返回空"""

import asyncio
import sys
sys.path.append(".")

from loguru import logger
from uuid import uuid4

from src.config import config
from src.models.knowledge import KnowledgeGraph, KnowledgeNode, OperationGuide, NodeType
from src.services.embedding_service import EmbeddingService
from src.knowledge.rag_service import RAGService


# 配置日志
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


async def test_rag_empty_knowledge_base():
    """测试1: 验证空知识库的RAG搜索行为"""
    print("\n" + "=" * 60)
    print("测试1: 空知识库RAG搜索")
    print("=" * 60)
    
    embedding = EmbeddingService()
    await embedding.initialize()
    
    knowledge_graph = KnowledgeGraph()
    rag = RAGService()
    await rag.initialize(embedding, knowledge_graph)
    
    # 检查知识库状态
    print(f"\n📊 知识库状态:")
    print(f"   - 指南数量: {len(knowledge_graph._guides)}")
    print(f"   - 节点数量: {len(knowledge_graph._nodes)}")
    print(f"   - 图节点数: {len(knowledge_graph._graph.nodes)}")
    
    # 尝试搜索
    query = "我想看新闻"
    print(f"\n🔍 搜索查询: '{query}'")
    
    result = await rag.retrieve(query, top_k=5, min_score=0.5)
    
    print(f"\n📋 搜索结果:")
    print(f"   - 找到指南: {len(result.guides)}")
    print(f"   - 找到节点: {len(result.nodes)}")
    print(f"   - 置信度: {result.confidence:.3f}")
    print(f"   - 上下文: {result.context[:100] if result.context else '(空)'}")
    
    if not result.guides and not result.nodes:
        print("\n⚠️ 结论: 知识库为空，所以RAG搜索返回空结果！")
        print("   解决方案: 需要先向知识库添加数据")
    
    await embedding.close()


async def test_rag_with_sample_data():
    """测试2: 添加示例数据后的RAG搜索"""
    print("\n" + "=" * 60)
    print("测试2: 添加示例数据后RAG搜索")
    print("=" * 60)
    
    embedding = EmbeddingService()
    await embedding.initialize()
    
    knowledge_graph = KnowledgeGraph()
    rag = RAGService()
    await rag.initialize(embedding, knowledge_graph)
    
    # 添加示例操作指南
    print("\n📝 添加示例数据...")
    
    sample_guides = [
        OperationGuide(
            id=uuid4(),
            title="如何使用浏览器看新闻",
            app_name="浏览器",
            feature_name="访问网站",
            steps=[
                "打开浏览器",
                "在地址栏输入网址",
                "按回车键访问",
                "浏览新闻内容"
            ],
            friendly_steps=[
                "找到浏览器图标，点一下打开",
                "在最上面的输入框里输入网址",
                "按键盘上的回车键",
                "就可以看新闻了"
            ],
            faq={"找不到浏览器怎么办": "在桌面或开始菜单找蓝色的e图标或圆形彩色图标"},
            quality_score=0.9
        ),
        OperationGuide(
            id=uuid4(),
            title="微信发送图片教程",
            app_name="微信",
            feature_name="发送图片",
            steps=[
                "打开微信",
                "选择联系人",
                "点击加号",
                "选择图片发送"
            ],
            friendly_steps=[
                "找到绿色的微信图标点开",
                "找到要发图片的人点进去",
                "点右下角的加号",
                "点相册选图片发送"
            ],
            faq={"加号在哪里": "在聊天界面的右下角"},
            quality_score=0.85
        ),
        OperationGuide(
            id=uuid4(),
            title="如何打开人民网",
            app_name="浏览器",
            feature_name="访问人民网",
            steps=[
                "打开浏览器",
                "输入www.people.com.cn",
                "按回车",
                "浏览人民网首页"
            ],
            friendly_steps=[
                "点开浏览器",
                "在地址栏输入人民网网址",
                "按回车键",
                "就能看到人民网了"
            ],
            faq={"人民网网址是什么": "www.people.com.cn"},
            quality_score=0.9
        ),
    ]
    
    # 添加示例知识节点
    sample_nodes = [
        KnowledgeNode(
            id=uuid4(),
            node_type=NodeType.APP,
            name="浏览器",
            description="用于访问网站、看新闻、搜索信息的应用程序",
            aliases=["上网", "网页", "IE", "Chrome", "Edge"]
        ),
        KnowledgeNode(
            id=uuid4(),
            node_type=NodeType.APP,
            name="微信",
            description="聊天、发消息、视频通话的应用",
            aliases=["WeChat", "绿色的", "聊天软件"]
        ),
        KnowledgeNode(
            id=uuid4(),
            node_type=NodeType.CONCEPT,
            name="新闻",
            description="查看最新资讯、时事新闻",
            aliases=["看新闻", "新闻网站", "资讯"]
        ),
    ]
    
    # 索引数据
    for guide in sample_guides:
        await rag.index_guide(guide)
        print(f"   ✅ 已索引指南: {guide.title}")
    
    for node in sample_nodes:
        await rag.index_node(node)
        print(f"   ✅ 已索引节点: {node.name}")
    
    # 检查知识库状态
    print(f"\n📊 知识库状态:")
    print(f"   - 指南数量: {len(knowledge_graph._guides)}")
    print(f"   - 节点数量: {len(knowledge_graph._nodes)}")
    print(f"   - 指南嵌入缓存: {len(rag._guide_embeddings)}")
    print(f"   - 节点嵌入缓存: {len(rag._node_embeddings)}")
    
    # 测试多个查询
    test_queries = [
        "我想看新闻",
        "怎么打开人民网",
        "微信怎么发图片",
        "如何上网",
        "打开浏览器",
    ]
    
    print("\n🔍 测试搜索:")
    for query in test_queries:
        result = await rag.retrieve(query, top_k=3, min_score=0.3)  # 降低阈值
        print(f"\n   查询: '{query}'")
        print(f"   - 找到指南: {len(result.guides)}")
        if result.guides:
            for g in result.guides:
                print(f"     • {g.title} (app: {g.app_name})")
        print(f"   - 找到节点: {len(result.nodes)}")
        if result.nodes:
            for n in result.nodes:
                print(f"     • {n.name}: {n.description[:30]}...")
        print(f"   - 置信度: {result.confidence:.3f}")
    
    await embedding.close()


async def test_embedding_service():
    """测试3: 验证Embedding服务是否正常工作"""
    print("\n" + "=" * 60)
    print("测试3: Embedding服务测试")
    print("=" * 60)
    
    embedding = EmbeddingService()
    await embedding.initialize()
    
    test_texts = [
        "我想看新闻",
        "打开浏览器访问人民网",
        "微信发送图片",
    ]
    
    print("\n📊 Embedding测试:")
    embeddings = []
    for text in test_texts:
        emb = await embedding.embed_text(text)
        embeddings.append(emb)
        print(f"   '{text}' -> 向量维度: {len(emb)}, 前5维: {emb[:5]}")
    
    # 计算相似度
    print("\n📊 相似度矩阵:")
    for i, t1 in enumerate(test_texts):
        for j, t2 in enumerate(test_texts):
            if i < j:
                sim = embedding.cosine_similarity(embeddings[i], embeddings[j])
                print(f"   '{t1[:10]}...' vs '{t2[:10]}...' = {sim:.3f}")
    
    await embedding.close()


async def main():
    """运行所有测试"""
    print("🚀 RAG服务调试测试")
    print("=" * 60)
    
    try:
        # 测试1: 空知识库
        await test_rag_empty_knowledge_base()
        
        # 测试2: 有数据的知识库
        await test_rag_with_sample_data()
        
        # 测试3: Embedding服务
        await test_embedding_service()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
        
        print("\n📋 诊断结论:")
        print("   1. RAG搜索返回空是因为知识库没有预置数据")
        print("   2. 需要在应用启动时加载示例数据或从视频提取知识")
        print("   3. 建议: 在 app_desktop.py 初始化时添加示例知识数据")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
