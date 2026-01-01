"""命令行界面"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from .agent.elderly_agent import ElderlyAssistantAgent, AgentConfig, AgentState
from .models.session import UserProfile


async def interactive_mode() -> None:
    """交互模式"""
    print("=" * 50)
    print("老年人电脑助手 - 交互模式")
    print("=" * 50)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("-" * 50)
    
    # 创建Agent
    agent = ElderlyAssistantAgent(AgentConfig(
        auto_execute=False,  # 需要用户确认每一步
    ))
    
    # 设置回调
    def on_speak(text: str) -> None:
        print(f"\n🔊 助手: {text}\n")
    
    def on_state_change(state: AgentState) -> None:
        state_emoji = {
            AgentState.IDLE: "😴",
            AgentState.LISTENING: "👂",
            AgentState.UNDERSTANDING: "🤔",
            AgentState.PLANNING: "📝",
            AgentState.EXECUTING: "⚡",
            AgentState.WAITING_USER: "⏳",
            AgentState.ERROR_RECOVERY: "🔧",
        }
        print(f"[状态: {state_emoji.get(state, '❓')} {state.value}]")
    
    agent.set_callbacks(
        on_speak=on_speak,
        on_state_change=on_state_change,
    )
    
    try:
        await agent.initialize()
        
        # 设置示例用户画像
        profile = UserProfile(
            name="张大爷",
            family_mapping={
                "老二": "张小明",
                "闺女": "张小红",
            },
            frequent_contacts=["张小明", "张小红", "李阿姨"],
            preferred_voice_speed=0.8,
        )
        agent.set_user_profile(profile)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit", "退出"):
                    print("\n再见！祝您生活愉快！")
                    break
                
                if user_input.lower() == "help":
                    print_help()
                    continue
                
                await agent.process_text_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n收到中断信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"处理输入时出错: {e}")
                print(f"\n抱歉，出了点问题: {e}")
    
    finally:
        await agent.close()


def print_help() -> None:
    """打印帮助信息"""
    print("""
帮助信息:
---------
您可以用自然语言告诉我您想做什么，比如：
  - "我想给女儿打个电话"
  - "帮我打开微信"
  - "我想看看老二发的照片"
  - "屏幕上有个东西关不掉"

特殊命令：
  - help  - 显示此帮助
  - quit  - 退出程序
  - exit  - 退出程序
""")


def main() -> None:
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>",
    )
    
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
