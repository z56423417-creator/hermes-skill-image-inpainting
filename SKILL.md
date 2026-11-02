---
name: image-inpainting
description: "照片局部重绘去水印/去杂物：掩膜→LaMa→只合成掩膜内像素。"
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [image, inpainting, watermark, lama, opencv]
    related_skills: [media-use, segment-anything-model]
---

# 图像局部重绘（去水印 / 去掉某个物体 / 擦掉文字）

## When to Use

用户发来一张图说「把水印抹掉」「把这个人去掉」「把这行字擦掉」时走本流程。目标是**像素级看不出痕迹**，不是打码、模糊或裁切；打码式处理只有在用户明说「看不清就行」时才算合格。

## 方案选择

| 场景 | 做法 |
| --- | --- |
| 默认（水印文字、贴纸、小物体，背景是虚化/纹理/布料） | LaMa（`simple-lama-inpainting`，torch CPU），几乎无缝 |
| 极细线条 + 纯色均匀背景，想几十秒出结果 | `cv2.inpaint(mask, 7, cv2.INPAINT_TELEA)` 兜底；大面积或跨强边缘会糊 |
| 水印压在图角、且重构图（裁切/平移）不损失内容 | 直接裁掉，零瑕疵，优先于任何重绘 |

## 流程

**0. 环境**（第三方脚本用兜底解释器，别往 hermes venv 装包）

```bash
PY="C:/Users/IvanZ/AppData/Local/Programs/Python/Python311/python.exe"
"$PY" -m pip install --index-url https://pypi.org/simple simple-lama-inpainting
```

首次运行自动下载 `big-lama.pt`（约 196 MB）到 `~/.cache/torch/hub/checkpoints/`，之后离线可用。CPU 上 2560×1912 约 110 s（模型加载另 23 s）——一次跑全图，别分块推理。

**1. 定位 + 看清背景**

先裁水印所在条带（如底部 20%）→ `vision_analyze` 让它报像素框和背景描述 → 再把该框 2 倍放大看一次（背景是虚化桌布？深色西装？有强边缘/接缝？）。坐标换算：vision 返回的是缩放后坐标，按它提示的倍数还原，`缩放后 x + 裁切偏移 = 原图坐标`。

**2. 建掩膜（成败在这步）**

- ROI 只框住水印 + 少量 padding，绝不框整图。
- 白色水印用**形态学白顶帽**，不要用亮度阈值：`cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ellipse(37))`，阈值 ~28。原因：水印常是白字压在白色桌布/浅色衣领上，纯亮度阈值会把整片背景一起吞掉；顶帽只保留「比周围亮的细结构」。
- 连通域过滤 `area >= 40` 去噪点，然后 `dilate(ellipse 15)` 覆盖抗锯齿与阴影边缘。**不做这一步会留一圈白边。**
- 出红色 overlay（`addWeighted(roi,.45,ov,.55,0)`）给 `vision_analyze` 复核：「是否全覆盖、是否误伤背景」，确认后再推理。这一步省掉一次几分钟的返工。

**3. 重绘**

```python
out = np.array(SimpleLama()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask))[:, :, :3]
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
```

**4. 只合成掩膜内像素**

```python
final = img.copy(); final[mask > 0] = out[mask > 0]
```

掩膜外逐像素保留原图：不引入全图重编码损失，也才能向用户交代「只动了这块」。交付时报量化：mask 像素数 + 与原图的 `mean abs diff`（JPEG q96 下约 0.14）。

**5. 目视验收（三张，缺一不可）**

1) mask overlay；2) 重绘区 2 倍放大（查残影/halo/糊块）；3) 整图缩略（查有无区域被破坏）。任一张不合格 → 回第 2 步调阈值或加 dilate 重跑，别在成品上叠滤镜掩盖。

## 坑

- **`cv2.imwrite` 在中文/非 ASCII 路径下会静默失败**（写不出文件，`os.path.getsize` 才报错）：一律走 `ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,96]); open(p,'wb').write(buf.tobytes())`，并 `os.makedirs(..., exist_ok=True)`；写完读回校验 shape 再交付。
- 顶帽只对**亮于背景的细结构**有效。要抹深色/彩色物体，用外部掩膜（SAM 或其它分割结果）喂进来，别硬调阈值。
- ROI 压到白色物体（衬衫、桌布）时掩膜会覆盖它的一部分，正常——LaMa 会用周围同类纹理补回；但 ROI 别越过物体边界太远，否则边界会被「抹平」。
- 不要把整图 resize 变小后再 inpaint：笔画糊成一片，还原回去鬼影更大。全分辨率做。

## 交付

放 `C:/Users/IvanZ/Pictures/watermark-removed/`（视频才走 `D:\生产视频`）：JPG q96 供预览与 `MEDIA:` 内嵌，PNG 给无损。结尾按用户习惯给编号选项（换图继续 / 压体积 / 锐化调色 / 存成流程）。

批量或多张图时用 `scripts/remove_watermark.py`，别手抄一遍推理代码：

```bash
"$PY" <SKILL_DIR>/scripts/remove_watermark.py in.jpg X0 Y0 X1 Y1 outdir [--thresh 28] [--dilate 15] [--mask mask.png] [--method cv2]
```

它一次产出 `_mask.png` / `_mask_overlay.png` / `_no_wm.jpg` / `_no_wm.png` / `_check2x.png`，并打印 mask px 与 mean abs diff。
