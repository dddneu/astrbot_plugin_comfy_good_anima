"""Pydantic 结构化输出模型。

约束 LLM 出稿必须输出结构化字段,便于:
1. Python 侧做代码化硬约束检查(冲突/分离/伪造 tag)。
2. 三层 prompt 分离不被破坏。
3. args 字段完整可注入。
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VisualBrief(BaseModel):
    """视觉简报,从情境因果链拆解(对齐 SKILL §2.3)。"""

    subject: str = Field(description="角色名/原创主体/人数")
    scene_container: str = Field(description="花海/教室/街道/神社/抽象空间(用户给出后不可改写)")
    action_relation: str = Field(description="单人姿态或多人互动关系,必须来自情境因果链")
    camera: str = Field(description="close-up / upper body / cowboy shot / full body")
    view_angle: str = Field(description="eye-level / from above / from below / from side")
    canvas: tuple[int, int] = Field(description="(width, height) 画布尺寸")
    light_direction: str = Field(description="光源位置和类型(窗光/侧光/背光/顶光/环境漫射)")
    subject_ratio: str = Field(description="主体在画面中的大致占比")
    situation_cause_chain: str = Field(description="情境因果锁:起因→角色反应→可见后果→最抓眼球的瞬间(内部字段)")

    @field_validator("situation_cause_chain", mode="before")
    @classmethod
    def _coerce_chain(cls, v):
        """LLM 可能返回 dict(结构化因果),转成字符串。"""
        if isinstance(v, dict):
            # 拼成 "cause -> reaction -> consequence -> moment" 形式
            parts = []
            for key in ("cause", "reaction", "consequence", "moment"):
                if key in v and v[key]:
                    parts.append(str(v[key]))
            return " -> ".join(parts) if parts else json.dumps(v, ensure_ascii=False)
        return v

    @field_validator("canvas", mode="before")
    @classmethod
    def _coerce_canvas(cls, v):
        """LLM 可能返回 list,转 tuple。"""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        return v

    @field_validator(
        "subject", "scene_container", "action_relation", "camera",
        "view_angle", "light_direction", "subject_ratio",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, v):
        """LLM 可能返回数字/None,统一转字符串。"""
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, float) and v < 1:
            # subject_ratio 可能是 0.7 这种,转成 "70%" 形式
            return f"{int(v * 100)}%"
        return str(v)


class ThreeLayerPrompt(BaseModel):
    """三层分离 prompt(对齐 SKILL §7-8)。"""

    hard_tags: list[str] = Field(description="质量/年代/安全/人数/角色/作品/画师/confirmed 外观。离散 tag,无完整句子")
    soft_phrases: list[str] = Field(description="动作/情感/环境效果短语。不查 Danbooru,不作 hard anchor")
    nltags_block: str = Field(description="空间布局/动作归属/接触/视线/遮挡/光源/景深/因果。有语法结构的连续描述")

    @field_validator("hard_tags", "soft_phrases", mode="before")
    @classmethod
    def _coerce_tag_list(cls, v):
        """LLM 可能返回逗号分隔字符串,拆成 list。"""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if v is None:
            return []
        return v

    @field_validator("nltags_block", mode="before")
    @classmethod
    def _coerce_nltags(cls, v):
        """nltags 可能为 list(逐句),拼成字符串。"""
        if isinstance(v, list):
            return " ".join(str(s).strip() for s in v if str(s).strip())
        return v or ""

    def assemble(self) -> str:
        """组装为 prompt_11: hard_tags + soft_phrases + nltags_block。"""
        parts: list[str] = []
        parts.append(", ".join(self.hard_tags))
        if self.soft_phrases:
            parts.append(", ".join(self.soft_phrases))
        if self.nltags_block:
            parts.append(self.nltags_block)
        return ", ".join(p for p in parts if p)


class AnimaArgs(BaseModel):
    """最终 args(对齐 comfyui-manager/SKILL.md §5 必含字段)。"""

    prompt_11: str = Field(description="正向 prompt = hard_tags + soft_phrases + nltags_block")
    prompt_12: str = Field(description="负向 prompt")
    width: int
    height: int
    batch_size: int = 5
    steps: int = Field(default=8, description="默认 8(turbo 模型低步数);不要随意拉高,30-40 步会让 turbo 过饱和")
    seed: Optional[int] = Field(default=None, description="None 时由 schema_injector 补随机")
    rtx_vsr_quality: str = "ULTRA"
    filename_prefix: str
    artist_chain: Optional[str] = None  # 仅 mixer workflow

    # FLSampler 排障参数( comfyui-manager/SKILL.md 排障表):默认不传,画质投诉时一次调一个
    fls_sharpness: Optional[float] = None      # 主体发糊 → 小幅提高(工作流默认 0.5)
    fls_fovea_strength: Optional[float] = None  # 纹理不足 → 小幅提高(默认 3.0)
    fls_mask_inertia: Optional[float] = None    # 焦点跳动 → 提高(默认 0.85)
    fls_cfg: Optional[float] = None             # turbo 保持 1.0;非 turbo 细节服从度弱才拉高
    fls_layer_filter: Optional[str] = None      # OUT=只在高频层注入,锁定底层大构图
    fls_step_decay: Optional[float] = None      # 步数衰减系数(0-1),前中期强引导后期自由生成

    # IP-Adapter 参数(仅 *-ref 工作流;LLM 可经 tune_params 工具调)
    ip_adapter_strength: Optional[float] = None         # 角色约束弱 → 提高(默认 1.0,范围 0-2);面部过强/衣服走样 → 降至 0.6-0.75
    ip_adapter_ref_image_size: Optional[int] = None     # 参考图边长(默认 512)
    ip_adapter_siglip_layer: Optional[int] = None       # SIGLIP 特征层(默认 -1)
    ip_adapter_ip_cfg_scale: Optional[float] = None     # IP-CFG 缩放(默认 4.0)
    ip_adapter_ip_cfg_separate: Optional[bool] = None   # IP-CFG 分离(默认 false)
    ip_adapter_use_lora: Optional[bool] = None         # 内置 LoRA(默认 true)
    ip_adapter_start_at: Optional[float] = None         # IP-Adapter 开始步数(0-1,默认 0.0)
    ip_adapter_end_at: Optional[float] = None          # IP-Adapter 结束步数(0-1,默认 0.45);降低让 IP-Adapter 尽早退场,把后期交给 InstantReferenceLoRA
    ip_adapter_layer_filter: Optional[str] = None      # OUT=只在高频层注入

    # Instant Reference 强度(仅 instantref 工作流;LLM 可经 tune_params 调,面板已无此配置)
    # 模型强度过高会把参考姿势焊进结果 → 姿态杂糅;角色细节弱时先降强度
    instantref_model_strength: Optional[float] = None   # 默认 1.2,范围 0-2(人物不像 → 提至 1.3-1.5)
    instantref_clip_strength: Optional[float] = None    # 默认 1.35,范围 0-2(画风不像 → 提至 1.4-1.5)

    # 负面排斥词:利用 CFG 排斥力逼迫模型生成更复杂纹理(追加到负面提示词)
    # 用逗号分隔,如 "simplified, plain clothes, missing details, blurry"
    negative_repel: Optional[str] = None

    # AnimaArtistOptions(仅 artist-mixer 工作流;LLM 可经 tune_params 调)
    # 注意:官方稳定配置为全部关闭(0/false),激进值会导致糊/人体杂糅,仅在用户明确要调试融合时用
    artist_ema_alpha: Optional[float] = None        # 默认 0.0
    artist_lowrank_k: Optional[int] = None          # 默认 1
    artist_static_capture: Optional[bool] = None    # 默认 false
    artist_anchor_q: Optional[bool] = None          # 默认 false

    # 参考图炼丹参数(仅 instantref 工作流;LLM 经 args 设置,程序化注入
    # ReferenceTaggingOptions / ReferenceTrainOptions 节点——临时 LoRA 的数据清洗与训练强度)
    # ⚠️ 打标悖论:exclude 只允许身份特征;衣服/动作/背景进 exclude 会被烤进角色,换装脱不下来
    ref_tag_exclude: Optional[str] = None            # 要焊进角色的身份 tag(1girl/solo/looking at viewer/发色/瞳色)
    ref_tag_prepend: Optional[str] = None            # 画风词前置(tagger 打标时强倾向;用户没提改画风时填 [wd14] 技法 tag)
    ref_tag_append: Optional[str] = None             # 画风词追加
    ref_tag_general_threshold: Optional[float] = None    # tagger 通用阈值(默认 0.35;角色细节极复杂 → 0.25)
    ref_tag_character_threshold: Optional[float] = None  # tagger 角色阈值(默认 0.85)
    ref_train_network_dim: Optional[int] = None      # 训练网络维度/rank(0=自动;复杂首饰/纹理 → 64~128)
    ref_train_steps: Optional[int] = None            # 训练步数(0=默认;强烈画风转换/细节极多 → 150~200)

    # Edit 模式字段(ICLoRAConcat 分屏编辑;LLM 填槽 + Python 组装)
    # left_anchor/right_edit 由 LLM 填槽,Python assemble_edit_prompt() 拼接为 prompt_2
    left_anchor: Optional[str] = None    # 左侧原图锚点(客观描述)
    right_edit: Optional[str] = None     # 右侧新状态(灵活描述)
    negative_tags: Optional[str] = None  # 负向 tag(LLM 填槽,Python 拼接)
    character_dna_tags: Optional[str] = None  # LLM 提纯的角色 DNA 标签（发色/瞳色/面部特征等核心身份）
    edited_tags: Optional[str] = None  # LLM 提取的修改/新增离散 tag，Python 端自动加权重 (tag:1.1)
    style_modifiers: Optional[str] = None  # 画风/画师/全局光影尾缀
    # Edit 模式专用:prompt_2 = 组装后的分屏正向 prompt(注入 ComfyUI node 2)
    # prompt_3 = 组装后的分屏负向 prompt(注入 ComfyUI node 3)
    # 普通模式下 prompt_2/prompt_3 未使用,统一用 prompt_11/prompt_12
    prompt_2: Optional[str] = None
    prompt_3: Optional[str] = None

    def to_args_dict(self) -> dict:
        d = self.model_dump()
        return {k: v for k, v in d.items() if v is not None}
