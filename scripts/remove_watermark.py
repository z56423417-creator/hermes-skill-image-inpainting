#!/usr/bin/env python
"""去水印 / 去物体：掩膜 + LaMa 重绘，只合成掩膜内像素。

依赖（装在兜底解释器里，不要用 hermes venv）:
  pip install --index-url https://pypi.org/simple simple-lama-inpainting

例:
  python remove_watermark.py in.jpg 740 1740 1830 1880 out
  python remove_watermark.py in.jpg 0 0 400 80 out --thresh 22 --dilate 21
  python remove_watermark.py in.jpg 0 0 400 80 out --mask mymask.png   # 深色/彩色物体，白=要抹
  python remove_watermark.py in.jpg 0 0 400 80 out --method cv2       # 秒级兜底

产出（outdir 下）: <stem>_mask.png / <stem>_mask_overlay.png /
                   <stem>_no_wm.jpg(96) / <stem>_no_wm.png / <stem>_check2x.png
并打印 mask 像素数与 mean abs diff。
"""
import argparse
import os
import sys

import cv2
import numpy as np


def imwrite_u(path, img, jpeg_quality=None):
    """Unicode-safe write: cv2.imwrite fails silently on non-ASCII paths on Windows."""
    ext = os.path.splitext(path)[1].lower() or '.png'
    if ext == '.jpeg':
        ext = '.jpg'
    params = []
    if ext == '.jpg' and jpeg_quality:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError('encode failed: %s' % path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return path


def build_mask(roi_gray, thresh=28.0, dilate=15, ksize=37, min_area=40):
    """White top-hat: keeps thin structures brighter than their surroundings.

    Works when the watermark sits on white cloth / pale clothing, where a plain
    brightness threshold would swallow the whole background.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    th = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
    m = (th > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    m = keep * 255
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('x0', type=int)
    ap.add_argument('y0', type=int)
    ap.add_argument('x1', type=int)
    ap.add_argument('y1', type=int)
    ap.add_argument('outdir')
    ap.add_argument('--thresh', type=float, default=28.0)
    ap.add_argument('--dilate', type=int, default=15)
    ap.add_argument('--ksize', type=int, default=37)
    ap.add_argument('--mask', help='external mask PNG, white = remove (skips top-hat)')
    ap.add_argument('--method', choices=['lama', 'cv2'], default='lama')
    a = ap.parse_args()

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit('cannot read %s' % a.image)
    h, w = img.shape[:2]
    x0, y0 = max(0, a.x0), max(0, a.y0)
    x1, y1 = min(w, a.x1), min(h, a.y1)
    if x1 <= x0 or y1 <= y0:
        sys.exit('bad ROI')
    roi = img[y0:y1, x0:x1]

    if a.mask:
        mm = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        if mm is None:
            sys.exit('cannot read mask %s' % a.mask)
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        mroi = (mm[y0:y1, x0:x1] > 127).astype(np.uint8) * 255
        if a.dilate > 0:
            mroi = cv2.dilate(mroi, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (a.dilate, a.dilate)))
    else:
        mroi = build_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                          a.thresh, a.dilate, a.ksize)

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = mroi
    npx = int((mask > 0).sum())
    if npx == 0:
        sys.exit('empty mask: lower --thresh, raise --ksize, or pass --mask')
    print('mask px=%d' % npx)

    stem = os.path.splitext(os.path.basename(a.image))[0]
    imwrite_u(os.path.join(a.outdir, stem + '_mask.png'), mask)
    ov = roi.copy()
    ov[mroi > 0] = (0, 0, 255)
    imwrite_u(os.path.join(a.outdir, stem + '_mask_overlay.png'),
              cv2.addWeighted(roi, 0.45, ov, 0.55, 0))

    if a.method == 'lama':
        from simple_lama_inpainting import SimpleLama
        pred = np.array(SimpleLama()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask))
        out = pred[:, :, :3]
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h))
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    else:
        out = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    final = img.copy()
    final[mask > 0] = out[mask > 0]
    diff = float(np.abs(final.astype(int) - img.astype(int)).mean())
    print('mean abs diff=%.2f' % diff)

    paths = [
        imwrite_u(os.path.join(a.outdir, stem + '_no_wm.jpg'), final, 96),
        imwrite_u(os.path.join(a.outdir, stem + '_no_wm.png'), final),
    ]
    chk = final[y0:y1, x0:x1]
    chk = cv2.resize(chk, (chk.shape[1] * 2, chk.shape[0] * 2),
                     interpolation=cv2.INTER_LANCZOS4)
    paths.append(imwrite_u(os.path.join(a.outdir, stem + '_check2x.png'), chk))
    for p in paths:
        print(p, os.path.getsize(p) // 1024, 'KB')


if __name__ == '__main__':
    main()
