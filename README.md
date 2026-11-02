# image-inpainting · 照片局部重绘去水印

[![CI](https://github.com/z56423417-creator/hermes-skill-image-inpainting/actions/workflows/ci.yml/badge.svg)](https://github.com/z56423417-creator/hermes-skill-image-inpainting/actions/workflows/ci.yml)

一个用于 [Hermes Agent](https://hermes-agent.nousresearch.com/docs) 的 skill：给一张图 + 一个矩形框，把框里的水印 / 贴纸 / 文字 / 小物体**重绘抹掉**，像素级看不出痕迹（不是打码、不是模糊、不是裁切）。

核心：**白顶帽建掩膜 → LaMa 重绘 → 只把掩膜内的像素贴回原图**。掩膜外逐像素保留原图，不引入全图重编码损失，也方便交代「只动了这一块」。

## 效果参考

2560×1912 的图，水印压在白桌布 + 深色中山装的交界处，CPU 上 LaMa 全分辨率推理约 110 s，掩膜外 `mean abs diff = 0`，整图级 `mean abs diff ≈ 0.14`（仅剩 JPEG 有损）。

## 安装

```bash
# 1) 依赖（建议装在独立的 Python 3.10+ 里，别污染其他 venv）
pip install --index-url https://pypi.org/simple simple-lama-inpainting
```

首次运行会自动下载 `big-lama.pt`（约 196 MB）到 `~/.cache/torch/hub/checkpoints/`，之后可离线跑。纯 CPU 即可，不需要显卡。

```bash
# 2) 装成 Hermes skill（可选）
git clone https://github.com/<you>/hermes-skill-image-inpainting.git \
  ~/.hermes/skills/media/image-inpainting
```

目录名保持 `image-inpainting`，Hermes 下次启动就能在技能列表里看到它，之后说「把这张图的水印抹掉」就会自动走这套流程。

## 用法

不装 Hermes 也能当普通脚本用：

```bash
python scripts/remove_watermark.py in.jpg X0 Y0 X1 Y1 outdir
```

`X0 Y0 X1 Y1` 是水印所在矩形（稍微留点余量）。

```bash
# 调掩膜灵敏度：水印很淡调低阈值，背景噪点多调高
python scripts/remove_watermark.py in.jpg 740 1740 1830 1880 out --thresh 28 --dilate 15

# 抹深色 / 彩色物体：自己给外部掩膜（白=要抹），别硬调阈值
python scripts/remove_watermark.py in.jpg 0 0 400 80 out --mask mymask.png

# 秒级兜底：不装 torch，用 OpenCV Telea，细线条 + 均匀背景够用
python scripts/remove_watermark.py in.jpg 0 0 400 80 out --method cv2
```

产出在 `outdir` 下：

| 文件 | 用途 |
| --- | --- |
| `*_mask.png` | 生成的掩膜 |
| `*_mask_overlay.png` | 红色覆盖预览，**先看这张再决定要不要跑推理** |
| `*_no_wm.jpg` / `*_no_wm.png` | 成品（q96 预览 / 无损） |
| `*_check2x.png` | 重绘区 2 倍放大，查残影和 halo |

脚本会打印掩膜像素数和 `mean abs diff`，可直接写进交付说明。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--thresh` | 28 | 白顶帽阈值。水印越淡越低 |
| `--dilate` | 15 | 掩膜外扩像素，覆盖抗锯齿与阴影。**漏了会留一圈白边** |
| `--mask` | – | 外部掩膜路径，给了就跳过自动检测 |
| `--method` | `lama` | `lama` 质量高但慢；`cv2` 秒级兜底 |

## 原理与几个坑

- **为什么用形态学白顶帽而不是亮度阈值**：水印经常是白字压在白色桌布、浅色衣领上，纯亮度阈值会把整片背景一起吞掉。顶帽（`MORPH_TOPHAT`，椭圆核 37）只保留「比周围亮的细结构」，正好是文字笔画。
- **必须 dilate**：文字边缘的抗锯齿和投影是半透明的，不外扩 10–15 px 会留一圈白边。
- **全分辨率推理**：先把图 resize 小再 inpaint，笔画会糊成一片，还原回去鬼影更大。
- **`cv2.imwrite` 在中文路径下静默失败**：Windows 上写不进去还不报错，要到 `os.path.getsize` 才炸。脚本里统一走 `cv2.imencode` + `open(...,'wb').write()`，并支持任意 Unicode 路径。
- **掩膜压到白色物体是正常的**：ROI 里如果有衬衫、桌布，掩膜会覆盖它一部分，LaMa 会用周围同类纹理补回；但 ROI 别越过物体边界太远，否则边界会被抹平。
- **强边缘交叉**：水印横跨两个差异很大的区域（如深色衣服 / 亮背景）时，LaMa 能接上但可能略显平滑，放大看若无结构问题即可接受；要求更高就缩 ROI 分两段跑。

## 局限

- 掩膜是矩形 ROI + 自动检测，**不支持交互式涂画**；形状复杂的目标请自己给 `--mask`。
- LaMa 是通用 inpainter，不保证还原真实结构（比如水印压在人脸上，会补出「合理但虚构」的皮肤）。
- 大 ROI（几百 px 见方）在 CPU 上会明显变慢，且更容易糊。

## 测试

CI（GitHub Actions，Ubuntu × py3.11/3.12 + Windows × py3.11 三档）跑 `tests/smoke_test.py`，覆盖：

| 检查 | 断言 |
| --- | --- |
| 产出 | 5 个文件齐全且非空 |
| 掩膜 | 合成水印笔画覆盖率 ≥ 97% |
| **核心承诺** | 掩膜外像素与输入**逐字节相同** |
| 效果 | ROI 内亮像素比例降到 0.2% 以下 |
| Unicode 路径 | 输出到中文目录不丢文件 |
| 外部掩膜 | `--mask` 逐像素照用，掩膜外不变 |
| 空掩膜 | 以非零码退出并给出可操作报错 |
| SKILL.md | frontmatter 的 name / description(≤60 字) / license 有效 |

本地跑：`pip install opencv-python-headless numpy && python tests/smoke_test.py`

LaMa 路径不进 CI（196 MB 权重 + CPU 约 2 分钟），需要时手动跑一次真图。

## 合规提示

只用于处理你自己拍摄 / 拥有版权的图片，或已获授权的内容。去掉他人作品的署名水印再传播通常侵犯著作权。

## License

MIT
