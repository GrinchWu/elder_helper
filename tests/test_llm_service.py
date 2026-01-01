"""LLM服务测试脚本 - 测试SimToM意图理解"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_simtom_intent():
    """测试SimToM意图理解 - 电脑操作场景"""
    from src.services.llm_service import LLMService, LLMConfig
    from src.models.session import UserProfile, TechLevel, CognitiveStyle
    
    # 配置 - 使用OpenAI兼容格式
    config = LLMConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen2.5-72B-Instruct",
    )
    
    # 创建详细的用户画像
    user_profile = UserProfile(
        name="王奶奶",
        age=68,
        tech_level=TechLevel.BEGINNER,
        cognitive_style=CognitiveStyle.VISUAL,
        family_mapping={
            "老大": "王明",
            "老二": "王华", 
            "闺女": "王丽",
            "老伴": "李爷爷",
        },
        frequent_contacts=["王明", "王华", "王丽", "张老师"],
        frequent_apps=["微信", "QQ", "浏览器"],
        interests=["看新闻", "养生知识", "戏曲"],
        known_skills=["打开微信", "发送文字消息", "接听视频通话"],
        misconceptions=["关闭窗口会丢失文件", "点错按钮会弄坏电脑"],
        anxiety_index=0.6,
        self_efficacy=0.4,
    )
    
    # 创建服务
    llm = LLMService(config)
    await llm.initialize()
    
    print("=" * 70)
    print("SimToM意图理解测试 - 电脑操作场景")
    print("=" * 70)
    print(f"\n用户画像：{user_profile.name}，{user_profile.age}岁")
    print(f"技术水平：{user_profile.tech_level.value}")
    print(f"家庭成员：{user_profile.family_mapping}")
    print("=" * 70)
    
    # 电脑操作测试用例
    test_cases = [
        # 发邮件场景
        "我想给老同事发个信",
        "怎么用电脑写信发给别人",
        "我要发个东西给张老师，就是那种正式一点的",
        
        # Word使用问题
        "我写的东西找不到了，刚才还在的",
        "那个写字的软件怎么把字变大",
        "我想把写好的东西打印出来",
        "怎么在文章里插一张照片",
        
        # 上网看新闻
        "我想看看人民网上有什么新消息",
        "怎么上网看新闻",
        "我想查查今天有什么大事",
        
        # 其他常见场景
        "屏幕上有脏东西关不掉",
        "电脑变得很慢怎么办",
        "我想看看我家老二",
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}: {test_input}")
        print("-" * 70)
        
        try:
            intent = await llm.understand_intent(
                user_input=test_input,
                user_profile=user_profile,
            )
            
            print(f"意图类型: {intent.intent_type.value}")
            print(f"标准化表述: {intent.normalized_text}")
            print(f"目标应用: {intent.target_app}")
            print(f"目标联系人: {intent.target_contact}")
            print(f"置信度: {intent.confidence.value:.2f}")
            
            if intent.parameters.get("specific_action"):
                print(f"具体操作: {intent.parameters['specific_action']}")
            
            # 打印SimToM分析（如果有）
            if intent.parameters.get("simtom_analysis"):
                analysis = intent.parameters["simtom_analysis"]
                if isinstance(analysis, dict) and "perspective_taking" in analysis:
                    pt = analysis["perspective_taking"]
                    if isinstance(pt, dict):
                        if "beliefs" in pt:
                            print(f"\nBDI分析 - Beliefs:")
                            beliefs = pt["beliefs"]
                            if isinstance(beliefs, dict):
                                print(f"  Known: {beliefs.get('known', [])[:2]}")
                                print(f"  Unknown: {beliefs.get('unknown', [])[:2]}")
                        if "desires" in pt:
                            print(f"BDI分析 - Desires:")
                            desires = pt["desires"]
                            if isinstance(desires, dict):
                                print(f"  Surface: {desires.get('surface_desire', '')}")
                                print(f"  Deep: {desires.get('deep_desire', '')}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    await llm.close()
    print("\n" + "=" * 70)
    print("测试完成")


async def test_specific_scenarios():
    """测试特定场景的详细分析"""
    from src.services.llm_service import LLMService, LLMConfig
    from src.models.session import UserProfile, TechLevel
    
    config = LLMConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen2.5-72B-Instruct",
    )
    
    user_profile = UserProfile(
        name="李爷爷",
        age=72,
        tech_level=TechLevel.NOVICE,
        family_mapping={"老伴": "王奶奶", "儿子": "李强"},
    )
    
    llm = LLMService(config)
    await llm.initialize()
    
    print("=" * 70)
    print("特定场景详细分析测试")
    print("=" * 70)
    
    scenarios = [
        {
            "name": "发邮件",
            "inputs": [
                "我想给老同事发个信",
                "怎么发邮件",
                "我要写封信发出去",
            ]
        },
        {
            "name": "Word问题",
            "inputs": [
                "我写的东西找不到了",
                "字太小看不清",
                "怎么保存我写的东西",
            ]
        },
        {
            "name": "上网看新闻",
            "inputs": [
                "我想看人民网",
                "怎么上网",
                "我要看新闻",
            ]
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"场景: {scenario['name']}")
        print("=" * 70)
        
        for user_input in scenario["inputs"]:
            print(f"\n输入: {user_input}")
            print("-" * 40)
            
            try:
                intent = await llm.understand_intent(user_input, user_profile)
                print(f"  意图: {intent.intent_type.value}")
                print(f"  应用: {intent.target_app}")
                print(f"  操作: {intent.parameters.get('specific_action', 'N/A')}")
                print(f"  置信度: {intent.confidence.value:.0%}")
            except Exception as e:
                print(f"  ❌ 错误: {e}")
    
    await llm.close()


async def interactive_test():
    """交互式测试"""
    from src.services.llm_service import LLMService, LLMConfig
    from src.models.session import UserProfile, TechLevel, CognitiveStyle
    
    config = LLMConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen2.5-72B-Instruct",
    )
    
    user_profile = UserProfile(
        name="测试用户",
        age=65,
        tech_level=TechLevel.BEGINNER,
        cognitive_style=CognitiveStyle.VISUAL,
        family_mapping={
            "老大": "张三",
            "老二": "李四",
            "闺女": "王五",
        },
        frequent_apps=["微信", "浏览器", "Word"],
        interests=["看新闻", "写文章"],
    )
    
    llm = LLMService(config)
    await llm.initialize()
    
    print("=" * 70)
    print("交互式SimToM测试")
    print("模拟老年人说话，测试意图理解")
    print("输入 'quit' 退出")
    print("=" * 70)
    print("\n示例输入：")
    print("  - 我想给老同事发个信")
    print("  - 我写的东西找不到了")
    print("  - 我想看看人民网上有什么新消息")
    print("  - 屏幕上有脏东西关不掉")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n请输入 (模拟老年人说话): ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue
            
            print("\n分析中...")
            intent = await llm.understand_intent(user_input, user_profile)
            
            print(f"\n📊 分析结果:")
            print(f"  意图类型: {intent.intent_type.value}")
            print(f"  标准化: {intent.normalized_text}")
            print(f"  目标应用: {intent.target_app or '无'}")
            print(f"  目标联系人: {intent.target_contact or '无'}")
            print(f"  置信度: {intent.confidence.value:.0%}")
            
            if intent.parameters.get("specific_action"):
                print(f"  具体操作: {intent.parameters['specific_action']}")
            
            if intent.parameters.get("clarification_question"):
                print(f"  ❓ 需要澄清: {intent.parameters['clarification_question']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    await llm.close()
    print("\n再见！")


def main():
    """主函数"""
    print("LLM服务测试 - SimToM意图理解")
    print("=" * 70)
    print("1. 电脑操作场景测试（发邮件、Word问题、上网看新闻等）")
    print("2. 特定场景详细分析")
    print("3. 交互式测试")
    print("=" * 70)
    
    choice = input("请选择测试项 (1/2/3): ").strip()
    
    if choice == "1":
        asyncio.run(test_simtom_intent())
    elif choice == "2":
        asyncio.run(test_specific_scenarios())
    elif choice == "3":
        asyncio.run(interactive_test())
    else:
        print("无效选择，运行默认测试...")
        asyncio.run(test_simtom_intent())


if __name__ == "__main__":
    main()
