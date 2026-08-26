"""Agent 双层决策层。

将 comfyui-animatool/SKILL.md 的硬约束沉淀为:
- prompts.py:出稿 System Prompt(模式注入 + 情境因果锁 + 三层分离)
- schemas.py:Pydantic 结构化输出模型
- draftsman.py:LLM 出稿层(意图→视觉简报→三层 prompt→args)
- react_agent.py:一次性出稿 Agent(多轮对话上下文)
- reviewer.py:代码化硬约束 + LLM 软约束自审
- pipeline.py:串联 draftsman → tag 校验 → reviewer → (不过则重出) → 注入 → 提交
- utils.py:公共工具函数(JSON 解析等)

双层设计:出稿负责创作,自审负责硬约束。硬约束走代码,软约束走 LLM。
"""

from anima_agent.agent.pipeline import AgentPipeline
from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief

__all__ = ["AgentPipeline", "AnimaArgs", "ThreeLayerPrompt", "VisualBrief"]
