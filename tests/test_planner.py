"""任务规划测试脚本 - 测试键盘输入需求 + 屏幕理解 + 任务规划"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_planner_with_vision():
    """测试任务规划：键盘输入需求 + 屏幕理解 + 任务规划"""
    from src.services.vision_service import VisionService, VLConfig
    from src.services.planner_service import PlannerService
    from src.models.intent import Intent, IntentType
    
    print("=" * 70)
    print("任务规划测试")
    print("=" * 70)
    print("1. 输入您的需求（例如：打开微信、打开浏览器搜索天气）")
    print("2. 系统会截取当前屏幕并分析")
    print("3. 根据屏幕内容和需求生成任务计划")
    print("输入 'quit' 退出")
    print("=" * 70)
    
    # 初始化Vision服务
    vl_config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(vl_config)
    await vision.initialize()
    
    # 初始化Planner服务
    planner = PlannerService()
    await planner.initialize()
    
    try:
        while True:
            print("\n" + "-" * 70)
            user_input = input("请输入您的需求: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if not user_input:
                print("请输入有效的需求")
                continue
            
            # 1. 截取并分析当前屏幕
            print("\n📸 正在截取屏幕...")
            screenshot, original_size = await vision.capture_screen()
            
            if not screenshot:
                print("❌ 截图失败")
                continue
            
            print(f"   屏幕尺寸: {original_size[0]}x{original_size[1]}")
            
            print("\n🔍 正在分析屏幕内容...")
            screen_analysis = await vision.analyze_screen(
                screenshot, 
                user_intent=user_input,
                original_size=original_size
            )
            
            # 显示屏幕分析结果
            print("\n📊 屏幕分析结果:")
            print(f"   应用: {screen_analysis.app_name or '未识别'}")
            print(f"   类型: {screen_analysis.screen_type or '未识别'}")
            print(f"   描述: {screen_analysis.description[:100] if screen_analysis.description else '无'}...")
            
            if screen_analysis.elements:
                print(f"\n   识别到 {len(screen_analysis.elements)} 个元素:")
                for i, elem in enumerate(screen_analysis.elements[:8], 1):
                    clickable = "🖱️" if elem.is_clickable else ""
                    text = elem.text or elem.description
                    print(f"     {i}. [{elem.element_type}] {text[:25] if text else '无'} {clickable}")
            
            # 2. 创建意图对象
            intent = Intent(
                raw_text=user_input,
                normalized_text=user_input,
                intent_type=IntentType.NAVIGATION,  # 导航/打开应用
            )
            
            # 3. 生成任务计划
            print("\n🧠 正在生成任务计划...")
            plan = await planner.create_plan(
                intent=intent,
                screen_analysis=screen_analysis,
            )
            
            # 显示任务计划
            print("\n" + "=" * 70)
            print("📋 任务计划")
            print("=" * 70)
            
            if plan.steps:
                for step in plan.steps:
                    print(f"\n步骤 {step.step_number}:")
                    print(f"  📝 描述: {step.description}")
                    print(f"  👴 指令: {step.friendly_instruction}")
                    if step.action:
                        print(f"  🎯 动作: {step.action.action_type.value}")
                        if step.action.element_description:
                            print(f"  🔘 目标: {step.action.element_description}")
                        if step.action.text:
                            print(f"  ⌨️ 输入: {step.action.text}")
                    if step.expected_result:
                        print(f"  ✅ 预期: {step.expected_result}")
                    if step.error_recovery_hint:
                        print(f"  ⚠️ 出错处理: {step.error_recovery_hint}")
            else:
                print("❌ 未能生成任务计划")
            
            print("\n" + "=" * 70)
            
    except KeyboardInterrupt:
        print("\n\n已中断")
    finally:
        await vision.close()
        await planner.close()


async def test_react_mode():
    """测试ReAct模式的任务规划"""
    from src.services.vision_service import VisionService, VLConfig
    from src.services.planner_service import PlannerService, PlannerContext
    from src.models.intent import Intent, IntentType
    
    print("=" * 70)
    print("ReAct模式任务规划测试")
    print("=" * 70)
    print("ReAct模式会逐步思考和执行，每一步都会观察结果")
    print("输入 'quit' 退出")
    print("=" * 70)
    
    # 初始化服务
    vl_config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(vl_config)
    await vision.initialize()
    
    planner = PlannerService()
    await planner.initialize()
    
    try:
        while True:
            print("\n" + "-" * 70)
            user_input = input("请输入您的需求: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 截取屏幕
            print("\n📸 正在截取屏幕...")
            screenshot, original_size = await vision.capture_screen()
            
            if not screenshot:
                print("❌ 截图失败")
                continue
            
            print("🔍 正在分析屏幕...")
            screen_analysis = await vision.analyze_screen(
                screenshot,
                user_intent=user_input,
                original_size=original_size
            )
            
            # 创建意图和上下文
            intent = Intent(
                raw_text=user_input,
                normalized_text=user_input,
                intent_type=IntentType.NAVIGATION,
            )
            
            context = PlannerContext(
                intent=intent,
                current_screen=screen_analysis,
                max_steps=10,
            )
            
            # ReAct循环
            print("\n" + "=" * 70)
            print("🤖 ReAct推理过程")
            print("=" * 70)
            
            for step_num in range(context.max_steps):
                print(f"\n--- 第 {step_num + 1} 步 ---")
                
                # 获取下一步建议
                react_step = await planner.suggest_next_action(context)
                
                print(f"💭 思考: {react_step.thought}")
                
                if react_step.action:
                    print(f"🎯 动作: {react_step.action.action_type.value}")
                    if react_step.action.element_description:
                        print(f"   目标: {react_step.action.element_description}")
                    if react_step.action.text:
                        print(f"   输入: {react_step.action.text}")
                    
                    # 检查是否完成
                    if react_step.action.action_type.value == "confirm":
                        print("\n✅ 任务规划完成！")
                        break
                
                # 模拟观察（实际应该执行动作后观察）
                react_step.observation = "等待执行..."
                context.history.append(react_step)
                
                # 询问是否继续
                cont = input("\n按Enter继续下一步，输入'stop'停止: ").strip()
                if cont.lower() == 'stop':
                    break
            
            print("\n" + "=" * 70)
            
    except KeyboardInterrupt:
        print("\n\n已中断")
    finally:
        await vision.close()
        await planner.close()


async def test_quick_plan():
    """快速测试：只生成计划，不进入交互模式"""
    from src.services.vision_service import VisionService, VLConfig
    from src.services.planner_service import PlannerService
    from src.models.intent import Intent, IntentType
    
    print("=" * 70)
    print("快速任务规划测试")
    print("=" * 70)
    
    # 预设的测试需求
    test_requests = [
        "打开微信",
        "打开浏览器搜索今天的天气",
        "打开记事本写一段文字",
    ]
    
    print("测试需求:")
    for i, req in enumerate(test_requests, 1):
        print(f"  {i}. {req}")
    
    choice = input("\n请选择 (1-3) 或输入自定义需求: ").strip()
    
    if choice in ["1", "2", "3"]:
        user_request = test_requests[int(choice) - 1]
    else:
        user_request = choice
    
    print(f"\n选择的需求: {user_request}")
    
    # 初始化服务
    vl_config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(vl_config)
    await vision.initialize()
    
    planner = PlannerService()
    await planner.initialize()
    
    try:
        # 截取并分析屏幕
        print("\n📸 截取屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        if not screenshot:
            print("❌ 截图失败")
            return
        
        print(f"   尺寸: {original_size[0]}x{original_size[1]}")
        
        print("\n🔍 分析屏幕...")
        screen_analysis = await vision.analyze_screen(
            screenshot,
            user_intent=user_request,
            original_size=original_size
        )
        
        print(f"   应用: {screen_analysis.app_name or '未识别'}")
        print(f"   元素数: {len(screen_analysis.elements)}")
        
        # 创建意图
        intent = Intent(
            raw_text=user_request,
            normalized_text=user_request,
            intent_type=IntentType.NAVIGATION,
        )
        
        # 生成计划
        print("\n🧠 生成任务计划...")
        plan = await planner.create_plan(
            intent=intent,
            screen_analysis=screen_analysis,
        )
        
        # 显示计划
        print("\n" + "=" * 70)
        print(f"📋 任务计划: {user_request}")
        print("=" * 70)
        
        if plan.steps:
            for step in plan.steps:
                print(f"\n【步骤 {step.step_number}】")
                print(f"  {step.friendly_instruction or step.description}")
                if step.action and step.action.element_description:
                    print(f"  → 点击: {step.action.element_description}")
        else:
            print("❌ 未能生成计划")
        
    finally:
        await vision.close()
        await planner.close()


def main():
    """主函数"""
    print("任务规划测试")
    print("=" * 70)
    print("1. 交互式任务规划（推荐）")
    print("2. ReAct模式测试")
    print("3. 快速测试")
    print("=" * 70)
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        asyncio.run(test_planner_with_vision())
    elif choice == "2":
        asyncio.run(test_react_mode())
    elif choice == "3":
        asyncio.run(test_quick_plan())
    else:
        print("默认运行交互式任务规划...")
        asyncio.run(test_planner_with_vision())


if __name__ == "__main__":
    main()
