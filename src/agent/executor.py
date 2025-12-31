"""动作执行器"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from loguru import logger

from ..models.action import Action, ActionType, ActionResult, ActionStatus


class ActionExecutor:
    """动作执行器 - 执行具体的UI操作"""
    
    def __init__(self) -> None:
        self._mouse_controller = None
        self._keyboard_controller = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化执行器"""
        try:
            # 延迟导入，避免在没有GUI的环境报错
            from pynput import mouse, keyboard
            self._mouse_controller = mouse.Controller()
            self._keyboard_controller = keyboard.Controller()
            self._initialized = True
            logger.info("ActionExecutor初始化完成")
        except ImportError:
            logger.warning("pynput未安装，将使用模拟模式")
            self._initialized = False
    
    async def execute(self, action: Action) -> ActionResult:
        """执行动作"""
        action.status = ActionStatus.EXECUTING
        
        try:
            if action.action_type == ActionType.CLICK:
                return await self._execute_click(action)
            elif action.action_type == ActionType.DOUBLE_CLICK:
                return await self._execute_double_click(action)
            elif action.action_type == ActionType.TYPE:
                return await self._execute_type(action)
            elif action.action_type == ActionType.SCROLL:
                return await self._execute_scroll(action)
            elif action.action_type == ActionType.WAIT:
                return await self._execute_wait(action)
            elif action.action_type == ActionType.BACK:
                return await self._execute_back(action)
            else:
                return ActionResult.fail(f"不支持的动作类型: {action.action_type}")
                
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            action.status = ActionStatus.FAILED
            return ActionResult.fail(str(e))
    
    async def _execute_click(self, action: Action) -> ActionResult:
        """执行点击"""
        if action.x is None or action.y is None:
            return ActionResult.fail("缺少点击坐标")
        
        if self._initialized and self._mouse_controller:
            try:
                from pynput.mouse import Button
                
                # 移动到目标位置
                self._mouse_controller.position = (action.x, action.y)
                await asyncio.sleep(0.1)  # 短暂延迟，让用户看到鼠标移动
                
                # 点击
                self._mouse_controller.click(Button.left, 1)
                
                action.status = ActionStatus.SUCCESS
                return ActionResult.ok(f"点击了 ({action.x}, {action.y})")
                
            except Exception as e:
                return ActionResult.fail(f"点击失败: {e}")
        else:
            # 模拟模式
            logger.info(f"[模拟] 点击 ({action.x}, {action.y})")
            action.status = ActionStatus.SUCCESS
            return ActionResult.ok(f"[模拟] 点击了 ({action.x}, {action.y})")
    
    async def _execute_double_click(self, action: Action) -> ActionResult:
        """执行双击"""
        if action.x is None or action.y is None:
            return ActionResult.fail("缺少点击坐标")
        
        if self._initialized and self._mouse_controller:
            try:
                from pynput.mouse import Button
                
                self._mouse_controller.position = (action.x, action.y)
                await asyncio.sleep(0.1)
                self._mouse_controller.click(Button.left, 2)
                
                action.status = ActionStatus.SUCCESS
                return ActionResult.ok(f"双击了 ({action.x}, {action.y})")
                
            except Exception as e:
                return ActionResult.fail(f"双击失败: {e}")
        else:
            logger.info(f"[模拟] 双击 ({action.x}, {action.y})")
            action.status = ActionStatus.SUCCESS
            return ActionResult.ok(f"[模拟] 双击了 ({action.x}, {action.y})")
    
    async def _execute_type(self, action: Action) -> ActionResult:
        """执行输入"""
        if not action.text:
            return ActionResult.fail("缺少输入文本")
        
        if self._initialized and self._keyboard_controller:
            try:
                # 逐字符输入，模拟真实打字
                for char in action.text:
                    self._keyboard_controller.type(char)
                    await asyncio.sleep(0.05)  # 每个字符间隔50ms
                
                action.status = ActionStatus.SUCCESS
                return ActionResult.ok(f"输入了: {action.text}")
                
            except Exception as e:
                return ActionResult.fail(f"输入失败: {e}")
        else:
            logger.info(f"[模拟] 输入: {action.text}")
            action.status = ActionStatus.SUCCESS
            return ActionResult.ok(f"[模拟] 输入了: {action.text}")
    
    async def _execute_scroll(self, action: Action) -> ActionResult:
        """执行滚动"""
        if self._initialized and self._mouse_controller:
            try:
                # 计算滚动量
                scroll_amount = action.scroll_amount // 100
                
                if action.scroll_direction == "up":
                    self._mouse_controller.scroll(0, scroll_amount)
                elif action.scroll_direction == "down":
                    self._mouse_controller.scroll(0, -scroll_amount)
                elif action.scroll_direction == "left":
                    self._mouse_controller.scroll(-scroll_amount, 0)
                elif action.scroll_direction == "right":
                    self._mouse_controller.scroll(scroll_amount, 0)
                
                action.status = ActionStatus.SUCCESS
                return ActionResult.ok(f"向{action.scroll_direction}滚动了")
                
            except Exception as e:
                return ActionResult.fail(f"滚动失败: {e}")
        else:
            logger.info(f"[模拟] 滚动: {action.scroll_direction}")
            action.status = ActionStatus.SUCCESS
            return ActionResult.ok(f"[模拟] 向{action.scroll_direction}滚动了")
    
    async def _execute_wait(self, action: Action) -> ActionResult:
        """执行等待"""
        wait_seconds = action.wait_ms / 1000.0
        await asyncio.sleep(wait_seconds)
        
        action.status = ActionStatus.SUCCESS
        return ActionResult.ok(f"等待了 {wait_seconds} 秒")
    
    async def _execute_back(self, action: Action) -> ActionResult:
        """执行返回"""
        if self._initialized and self._keyboard_controller:
            try:
                from pynput.keyboard import Key
                
                # 按下Alt+Left (Windows返回)
                self._keyboard_controller.press(Key.alt)
                self._keyboard_controller.press(Key.left)
                self._keyboard_controller.release(Key.left)
                self._keyboard_controller.release(Key.alt)
                
                action.status = ActionStatus.SUCCESS
                return ActionResult.ok("返回上一页")
                
            except Exception as e:
                return ActionResult.fail(f"返回失败: {e}")
        else:
            logger.info("[模拟] 返回上一页")
            action.status = ActionStatus.SUCCESS
            return ActionResult.ok("[模拟] 返回上一页")
    
    async def execute_with_tolerance(
        self,
        action: Action,
        tolerance_radius: int = 10,
    ) -> ActionResult:
        """带容错的执行 - 处理老年人手抖问题"""
        if action.action_type not in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
            return await self.execute(action)
        
        # 对于点击操作，增加点击区域的容错
        # 实际实现中可以通过多次尝试或扩大点击区域来实现
        
        # 这里简单实现：如果第一次失败，尝试在周围区域点击
        result = await self.execute(action)
        
        if not result.success and action.x and action.y:
            # 尝试在周围点击
            offsets = [(0, 0), (-5, 0), (5, 0), (0, -5), (0, 5)]
            for dx, dy in offsets:
                action.x += dx
                action.y += dy
                result = await self.execute(action)
                if result.success:
                    break
                action.x -= dx
                action.y -= dy
        
        return result
    
    async def execute_with_confirmation(
        self,
        action: Action,
        confirm_callback,
    ) -> ActionResult:
        """需要确认的执行"""
        # 先询问用户确认
        confirmed = await confirm_callback(action.to_friendly_description())
        
        if not confirmed:
            action.status = ActionStatus.CANCELLED
            return ActionResult(
                success=False,
                message="用户取消了操作",
            )
        
        return await self.execute(action)
