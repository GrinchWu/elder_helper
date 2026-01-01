"""简化的Agent运行入口 - 打通完整流程（不含语音输出）"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

from loguru import logger

from .config import config
from .models.intent import Intent, IntentType
from .models.task import Task, TaskStatus, TaskPlan
from .models.session import UserProfile
from .models.knowledge import KnowledgeGraph
from .services.llm_service import LLMService
from .services.vision_service import VisionService, VLConfig, ScreenAnalysis
from .services.planner_service import PlannerService
from .services.safety_service import SafetyService
from .services.executor_service import ExecutorService
from .services.embedding_service import EmbeddingService
from .knowledge.rag_service import RAGService


class SimpleElderlyAgent:
    """简化版老年人助手Agent - 用于测试完整流程"""
    
    def __init__(self):
        self._llm: LLMService = None
        self._vision: VisionService = None
        self._planner: PlannerService = None
        self._safety: SafetyService = None
        self._executor: ExecutorService = None
        self._embedding: EmbeddingService = None
        self._rag: RAGService = None
        self._knowledge_graph: KnowledgeGraph = None
        
        self._user_profile: UserProfile = None
    
    async def initialize(self):
        """初始化所有服务"""
        print("=" * 60)
        print("🤖 老年人电脑助手 - 初始化中...")
        print("=" * 60)
        
        # 初始化 LLM 服务
        print("  📝 初始化意图理解服务...")
        self._llm = LLMService()
        await self._llm.initialize()
        
        # 初始化 Vision 服务
        print("  👁️ 初始化视觉服务...")
        vl_config = VLConfig(
            api_key=config.api.api_key,
            model_light=config.api.vl_model_light,
            model_heavy=config.api.vl_model_heavy,
        )
        self._vision = VisionService(vl_config)
        await self._vision.initialize()
        
        # 初始化 Planner 服务
        print("  📋 初始化规划服务...")
        self._planner = PlannerService()
        await self._planner.initialize()
        
        # 初始化 Safety 服务
        print("  🛡️ 初始化安全服务...")
        self._safety = SafetyService()
        
        # 初始化 Embedding 服务
        print("  🔢 初始化向量嵌入服务...")
        self._embedding = EmbeddingService()
        await self._embedding.initialize()
        
        # 初始化知识图谱
        print("  📚 初始化知识图谱...")
        self._knowledge_graph = KnowledgeGraph()
        
        # 初始化 RAG 服务
        print("  🔍 初始化RAG检索服务...")
        self._rag = RAGService()
        await self._rag.initialize(
            embedding_service=self._embedding,
            knowledge_graph=self._knowledge_graph,
        )
        
        # 将 RAG 服务关联到 Planner
        self._planner.set_rag_service(self._rag)
        
        # 初始化 Executor 服务
        print("  ⚡ 初始化执行服务...")
        self._executor = ExecutorService()
        # 关联外部服务，避免重复初始化
        self._executor.set_vision_service(self._vision)
        self._executor.set_planner_service(self._planner)
        await self._executor.initialize()
        
        # 设置执行器回调
        self._executor.set_callbacks(
            on_status_update=lambda msg: print(f"  {msg}"),
            on_ask_user=lambda q: print(f"\n❓ {q}\n"),
        )
        
        # 设置默认用户画像
        self._user_profile = UserProfile(
            name="用户",
            family_mapping={
                "老二": "张小明",
                "闺女": "张小红",
            },
            frequent_contacts=["张小明", "张小红"],
        )
        
        print("\n✅ 所有服务初始化完成！")
        print("-" * 60)
    
    async def close(self):
        """关闭所有服务"""
        print("\n正在关闭服务...")
        if self._llm:
            await self._llm.close()
        if self._vision:
            await self._vision.close()
        if self._planner:
            await self._planner.close()
        if self._executor:
            await self._executor.close()
        if self._embedding:
            await self._embedding.close()
        print("服务已关闭")
    
    async def process_input(self, user_input: str) -> bool:
        """
        处理用户输入 - 完整流程
        
        流程:
        1. 安全检查
        2. 意图理解 (LLM)
        3. 截屏分析 (Vision Layer 1)
        4. 任务规划 (Planner)
        5. 任务执行 (Executor)
        
        返回: True 继续, False 退出
        """
        if not user_input.strip():
            return True
        
        if user_input.lower() in ("quit", "exit", "退出", "q"):
            print("\n👋 再见！祝您生活愉快！")
            return False
        
        if user_input.lower() in ("help", "帮助", "h"):
            self._print_help()
            return True
        
        print(f"\n{'='*60}")
        print(f"📥 收到输入: {user_input}")
        print(f"{'='*60}")
        
        try:
            # ========== 1. 安全检查 ==========
            print("\n🛡️ [步骤1] 安全检查...")
            safety_result = self._safety.check_text_safety(user_input)
            if not safety_result.is_safe:
                print(f"  ⚠️ 安全警告: {safety_result.warnings}")
                print(f"  💡 建议: {safety_result.suggestions}")
                if safety_result.blocked_reason:
                    print(f"  ❌ 操作被阻止: {safety_result.blocked_reason}")
                    return True
            else:
                print("  ✅ 安全检查通过")
            
            # ========== 2. 意图理解 ==========
            print("\n🧠 [步骤2] 理解您的意图...")
            intent = await self._llm.understand_intent(
                user_input=user_input,
                user_profile=self._user_profile,
            )
            
            print(f"  📌 意图类型: {intent.intent_type.value}")
            print(f"  📝 标准化表述: {intent.normalized_text}")
            print(f"  🎯 目标应用: {intent.target_app or '未指定'}")
            print(f"  📊 置信度: {intent.confidence.value:.2f}")
            
            if intent.confidence.is_low:
                print(f"\n  ❓ 我不太确定您想做什么，能再说详细一点吗？")
                return True
            
            # ========== 3. 截屏分析 ==========
            print("\n👁️ [步骤3] 分析当前屏幕...")
            screenshot, original_size = await self._vision.capture_screen()
            
            if not screenshot:
                print("  ❌ 截屏失败")
                return True
            
            print(f"  📐 屏幕尺寸: {original_size[0]}x{original_size[1]}")
            
            # 使用轻量级模型分析页面状态
            screen_state = await self._vision.analyze_screen_state(
                screenshot,
                user_intent=intent.normalized_text or user_input,
            )
            
            print(f"  📱 当前应用: {screen_state.app_name}")
            print(f"  📄 页面状态: {screen_state.screen_state}")
            print(f"  📝 描述: {screen_state.description[:100]}..." if len(screen_state.description) > 100 else f"  📝 描述: {screen_state.description}")
            
            if screen_state.warnings:
                print(f"  ⚠️ 警告: {screen_state.warnings}")
            
            # 转换为兼容格式
            screen_analysis = ScreenAnalysis(
                app_name=screen_state.app_name,
                screen_type=screen_state.screen_state,
                description=screen_state.description,
                suggested_actions=[screen_state.suggested_action] if screen_state.suggested_action else [],
                warnings=screen_state.warnings,
            )
            
            # ========== 4. 任务规划 ==========
            print("\n📋 [步骤4] 生成任务计划...")
            plan = await self._planner.create_plan(
                intent=intent,
                screen_analysis=screen_analysis,
            )
            
            if not plan.steps:
                print("  ❌ 无法生成任务计划")
                print("  💡 抱歉，我不太确定该怎么帮您完成这个操作。您能再说详细一点吗？")
                return True
            
            print(f"  ✅ 已生成 {len(plan.steps)} 步计划:")
            for i, step in enumerate(plan.steps, 1):
                print(f"     {i}. {step.friendly_instruction or step.description}")
            
            # ========== 5. 确认执行 ==========
            print(f"\n{'='*60}")
            confirm = input("❓ 是否开始执行? (y/n): ").strip().lower()
            if confirm not in ("y", "yes", "是", "好", ""):
                print("  ⏹️ 已取消执行")
                return True
            
            # ========== 6. 任务执行 ==========
            print("\n⚡ [步骤5] 开始执行任务...")
            print("-" * 40)
            
            # 将已生成的计划传递给 Executor，避免重复规划
            task = await self._executor.execute_task(intent, plan=plan)
            
            print("-" * 40)
            if task.status == TaskStatus.COMPLETED:
                print("\n🎉 任务完成！")
            else:
                print(f"\n❌ 任务未完成，状态: {task.status.value}")
            
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n❌ 抱歉，出了点问题: {e}")
        
        return True
    
    def _print_help(self):
        """打印帮助信息"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    老年人电脑助手 - 帮助                      ║
╠══════════════════════════════════════════════════════════════╣
║  您可以用自然语言告诉我您想做什么，比如：                      ║
║                                                              ║
║  💬 "我想给女儿打个电话"                                      ║
║  💬 "帮我打开微信"                                            ║
║  💬 "我想看看老二发的照片"                                    ║
║  💬 "屏幕上有个东西关不掉"                                    ║
║  💬 "帮我打开浏览器看新闻"                                    ║
║                                                              ║
║  特殊命令：                                                   ║
║    help / 帮助  - 显示此帮助                                  ║
║    quit / 退出  - 退出程序                                    ║
╚══════════════════════════════════════════════════════════════╝
""")


async def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",  # 只显示警告和错误
        format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>",
    )
    
    agent = SimpleElderlyAgent()
    
    try:
        await agent.initialize()
        
        print("\n💡 输入 'help' 查看帮助，输入 'quit' 退出")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                should_continue = await agent.process_input(user_input)
                if not should_continue:
                    break
            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，正在退出...")
                break
            except EOFError:
                break
    
    finally:
        await agent.close()


def run():
    """入口函数"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    run()
