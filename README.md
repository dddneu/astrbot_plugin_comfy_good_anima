# Anima 画师 (astrbot_plugin_comfy_good_anima)

AstrBot 生图插件:把自然语言描述变成高质量动漫插画。不是简单转发提示词——
内置 **LLM 情境出稿**(情境因果链 → 三层 Prompt 分离)、**Danbooru 标签库校验**
(10 万+标签,不伪造 tag)、**ReAct 自纠错循环** 和 **连续修改**(继承 seed,只改你要求的部分)。

## 效果

```
/draw 教室窗边的银发少女,午后阳光          → 新图
/draw 头发换成金色                          → 修改上一张(同 seed,只换发色)
/draw 重抽                                  → 同描述换 seed
/draw 融合 wlop 和 krenz 的风格             → 画师融合工作流
/draw 这张太糊了,锐一点                     → 自动调 FLSampler 锐度参数重绘
/draw [附图] 换成校服                        → 参考图自动打标 + 快速 LoRA(Instant Reference),模型认识该角色(首图多等 1~3 分钟训练,相同参考图缓存)
/draw 参考图约束太弱 / 融合太糊             → LLM 经 tune_params 调参数
/draw 再画一张她在窗边的图                   → 会话角色记忆:一次对话内保持已认识角色的外观一致
/anima_status                               → 最近生成的 seed / prompt_id
/draw_check                                → 环境自检(见下)
```

单张耗时约 3~6 分钟(出稿 LLM + ComfyUI 生成)。生成期间机器人其他功能不受影响,
多人同时请求自动排队(同人按顺序、全局并发上限可配)。

> **LLM 可调参数(白名单+值域)**:用户抱怨画质/参考约束弱/融合不稳时,出稿 Agent
> 会经 `tune_params` 工具调参,一次一个、小幅调整、自动钳制合法范围:
> FLSampler(`fls_sharpness` 0-3 / `fls_fovea_strength` 0-6 / `fls_mask_inertia` 0-1)、
> IP-Adapter(`ip_adapter_strength` 0-2 / `ip_adapter_ref_image_size` 256-1024 /
> `ip_adapter_siglip_layer` -8~0 / `ip_adapter_ip_cfg_scale` 0-10 /
> `ip_adapter_ip_cfg_separate` 0/1 / `ip_adapter_use_lora` 0/1 /
> `ip_adapter_start_at` 0-1 / `ip_adapter_end_at` 0-1 / `ip_adapter_layer_filter` OUT 等)、
> ArtistMixer(`artist_ema_alpha` 0-1 / `artist_lowrank_k` 1-8 /
> `artist_static_capture` 0/1 / `artist_anchor_q` 0/1,稳定配置为全部关闭)。

> **参考图优化技巧**:IP-Adapter 步数截断(`ip_adapter_start_at`/`ip_adapter_end_at`)可让其在 0~0.5 步奠定基础后退出,把后期细节交给 InstantReferenceLoRA;分层过滤(`ip_adapter_layer_filter`=OUT)只在 OUT Blocks 注入,防止宏观构图被干扰。

## 安装

### 方式一:插件市场(推荐)
AstrBot WebUI → 插件市场 → 搜索「Anima 画师」→ 安装。

### 方式二:手动
```bash
cd AstrBot/data/plugins
git clone https://github.com/yourname/astrbot_plugin_comfy_good_anima
```
重启 AstrBot(首次启动会自动安装 Python 依赖)。

## 配置(插件面板)

| 配置项 | 默认 | 说明 |
|---|---|---|
| `comfyui_server` | `127.0.0.1:8188` | ComfyUI API 地址 (见 `_conf_schema.json` 配置) |

**LLM 不在本插件配置**:出稿用的聊天模型直接使用 AstrBot 的「服务提供商」配置
(WebUI → 配置页 → 服务提供商,任意 OpenAI 兼容 LLM 均可),且跟随用户在会话中切换的模型。
| `workflow` | aesthetic-lora | base(裸模型)/ aesthetic-lora(双 LoRA)/ artist-mixer(画师融合)/ instantref(快速 LoRA 参考图)/ instantref-ipadapter(IP-Adapter + LoRA 组合);附参考图时默认自动切 `aesthetic-lora-instantref` |
| `nsfw` | `false` | NSFW 模式;Anima 模型用 NSFW 数据训练,开启后画质显著提升。不安全模式也会自动拒绝 explicit 内容 |
| `ref_tagger` | `true` | 参考图自动打标:带参考图时先用 PixAI Booru Tagger 打标图中真实内容并注入 LLM 出稿(防 LLM 看不到图乱编 prompt)。需 ComfyUI-PixAI-Tagger;打标失败自动降级不阻断生图 |
| `instantref_model_strength` | `1.0` | Instant Reference 模型强度(0~2)。角色细节弱/姿态杂糅时降到 `0.5~0.7`(姿势焊死是单图 LoRA 通病,先降强度再提训练量) |
| `instantref_clip_strength` | `1.0` | Instant Reference CLIP 强度(0~2),一般保持默认 |
| `wait_for_image` | `true` | 等图模式;false 则提交即回、生成完主动推送 |
| `max_concurrent` | `3` | 全局并发生成上限 |
| `reply_quote` | `true` | 回复时引用触发消息(群里多人请求不混乱) |
| `send_progress` | `true` | 发送「正在构思...」进度提示 |



## ComfyUI 前置条件(重要)

插件只负责出稿和提交,**生图由你的 ComfyUI 执行**。装好后先发 `/draw_check` 自检。

**自定义节点**(ComfyUI Manager 搜索安装):

| 节点 | 用途 |
|---|---|
| AnimaBoosterLoader / AnimaTeaCache / FLS_SamplerV4 | Anima 采样链(随 Anima 模型发布) |
| RTXVideoSuperResolution | 超分(需要 **NVIDIA RTX 显卡**) |
| AnimaArtistPack 系列 | 仅画师融合工作流需要 |
| InstantReferenceLoRA(comfyui-instant-reference) | **参考图工作流(`*-instantref`)**:附参考图时在 ComfyUI 内快速训练 LoRA 并应用,让模型认识该角色(替代 IP-Adapter;需其 sd-scripts 依赖) |
| AnimaIPAdapterLoader / AnimaIPAdapterApply | 仅 `*-ref` 工作流需要（基于 IP-Adapter 的角色参考） |
| PixAI Booru Tagger | 参考图自动打标(默认开启,打标失败自动降级);缺了只影响参考图质量 |
| ResizeImagesByLongerEdge | 打标前缩放参考图(如 KJNodes 提供) |

> 参考图方案:默认附参考图 → 自动走 `*-instantref`(快速 LoRA)。角色细节弱/姿态杂糅时:面板调低 `instantref_model_strength`(如 0.6)。

**模型文件**(放入 ComfyUI `models/` 对应目录):

| 文件 | 类型 | 位置 |
|---|---|---|
| `anima-base-v1.0.safetensors` | checkpoint | checkpoints/ |
| `qwen_3_06b_base.safetensors` | text encoder | text_encoders/ (CLIPLoader) |
| `qwen_image_vae.safetensors` | VAE | vae/ |
| `anima-base-1-masterpiece-v51.safetensors` | LoRA | loras/ |
| `anima-highres-aesthetic-boost.safetensors` | LoRA | loras/ |
| `ip_adapter-Character_Reference-10.safetensors` | IP-Adapter | AnimaIPAdapterLoader 指定目录(参考图工作流必需,缺了 ref 提交会被拒) |

标签数据库(29MB sqlite)已随插件打包,无需额外下载。

## 故障排查

| 症状 | 处置 |
|---|---|
| 插件加载失败 | 看启动日志;依赖安装失败手动 `pip install -r requirements.txt` |
| `/draw_check` ComfyUI 连不通 | 确认 ComfyUI 已启动、地址端口正确(默认 8188) |
| 缺自定义节点/模型 | 按自检输出提示安装;RTX 超分节点需 N 卡 |
| `/draw` 报未获取到聊天模型 | 在 AstrBot WebUI → 服务提供商 配置 LLM 并启用 |
| 生成超时/无响应 | ComfyUI 队列是否堆积;看 AstrBot 日志中 prompt_id |
| 画质偏软/发糊 | `/draw 太糊了,锐一点` 插件会按排障表自动调采样器参数 |
| 画师融合出图糊/人体杂糅 | workflow 里 AnimaArtistOptions 的 stabilizer 参数被改成了激进值。默认配置(所有 stabilizer 关闭,即 `artist_ema_alpha=0`, `artist_static_capture=false`, `artist_anchor_q=false`, `lowrank_k=1`)是官方推荐稳定配置,不要改成 `0.95` / `4` / `0.5` 这类值 |

## 致谢与来源

本插件的生图方法论全部来自 **[ShiroEirin/comfyui-good-anima](https://github.com/ShiroEirin/comfyui-good-anima)** 项目——
情境因果锁、画面八维补全、三层 Prompt 分离、Danbooru 标签校验优先级、动态负向组装、
FLSampler 排障表等领域规约均沉淀自该项目的 SKILL 文档。

本插件是该项目「PowerShell + Rust + Node 工具链」向「AstrBot 常驻 Python Agent」的迁移实现:
原项目的 sqlite 标签库、workflow 定义原样复用,决策逻辑从 SKILL.md 文本约束转为
System Prompt + 程序化校验 + ReAct 工具循环。向原项目致谢。

## 许可

GPL-3.0(见 LICENSE),继承自上游 [comfyui-good-anima](https://github.com/ShiroEirin/comfyui-good-anima)。
标签数据来自 Anima 模型社区,仅供个人使用。
