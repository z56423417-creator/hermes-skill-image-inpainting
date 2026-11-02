#!/usr/bin/env python
"""End-to-end smoke test for scripts/remove_watermark.py (cv2 path only).

Runs on GitHub Actions with no torch: the LaMa path is exercised manually and
is not part of CI. What IS asserted here:

  1. the CLI produces all five artifacts
  2. pixels outside the mask are byte-identical to the input  (the core promise)
  3. the auto-built mask actually covers the synthetic watermark strokes
  4. the watermark is gone afterwards (no bright residue left in the ROI)
  5. unicode output paths work    (cv2.imwrite silently fails on Windows)
  6. an external --mask is honoured
  7. SKILL.md frontmatter stays valid (name / description / license)

Usage: python tests/smoke_test.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "remove_watermark.py")
TEXT = "DEMO 12345"


def fail(msg):
    print("FAIL: %s" % msg)
    sys.exit(1)


def ok(msg):
    print("  ok  %s" % msg)


def make_test_image(path):
    """Light 'blurred cloth' on the left, dark 'jacket' on the right, white text across both."""
    h, w = 420, 640
    yy, xx = np.mgrid[0:h, 0:w]
    base = 188.0 + 18.0 * np.sin(xx / 70.0) + 9.0 * np.cos(yy / 50.0)
    img = np.dstack([base, base * 0.99, base * 0.93]).astype(np.float32)
    img[:, 430:] = np.array([46.0, 39.0, 31.0], np.float32)  # navy panel
    img = cv2.GaussianBlur(img, (0, 0), 3)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.9, 4
    (tw, th), bl = cv2.getTextSize(TEXT, font, scale, thick)
    x, y = 40, 300
    cv2.putText(img, TEXT, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    noise = np.random.default_rng(7).normal(0, 2.5, (h, w, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

    pad = 24
    roi = (max(0, x - pad), max(0, y - th - bl - pad),
           min(w, x + tw + pad), min(h, y + bl + pad))
    return img, roi


def run(args, expect_code=0):
    p = subprocess.run([sys.executable, SCRIPT] + [str(a) for a in args],
                       capture_output=True, text=True)
    if p.returncode != expect_code:
        print(p.stdout)
        print(p.stderr)
        fail("exit %d (expected %d) for %s" % (p.returncode, expect_code, args))
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="wm-smoke-")
    try:
        print("[1] generate synthetic image")
        # PNG on purpose: the CLI must be compared byte-exactly, and JPEG is lossy.
        src = os.path.join(tmp, "in.png")
        img, (x0, y0, x1, y1) = make_test_image(src)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray[y0:y1, x0:x1] >= 245)
        if ink.sum() < 300:
            fail("synthetic watermark too small to test (%d px)" % ink.sum())
        ok("watermark ink px=%d, ROI=%s" % (int(ink.sum()), (x0, y0, x1, y1)))

        print("[2] cv2 path, auto mask")
        out = os.path.join(tmp, "out")
        p = run([src, x0, y0, x1, y1, out, "--method", "cv2"])
        for suffix in ("_mask.png", "_mask_overlay.png", "_no_wm.jpg",
                       "_no_wm.png", "_check2x.png"):
            f = os.path.join(out, "in" + suffix)
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                fail("missing/empty artifact: %s" % f)
        ok("5 artifacts written")

        mask = cv2.imread(os.path.join(out, "in_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (h, w):
            fail("mask size %s != image size %s" % (mask.shape[:2], (h, w)))
        if (mask > 0).sum() == 0:
            fail("empty mask")

        print("[3] mask covers the watermark strokes")
        cov = (mask[y0:y1, x0:x1] > 0)[ink].mean()
        if cov < 0.97:
            fail("mask covers only %.1f%% of ink pixels" % (100 * cov))
        ok("ink coverage %.2f%%" % (100 * cov))

        print("[4] pixels outside the mask are untouched")
        res = cv2.imread(os.path.join(out, "in_no_wm.png"), cv2.IMREAD_COLOR)
        if res is None or res.shape != img.shape:
            fail("output unreadable or wrong size")
        outside = mask == 0
        if not np.array_equal(res[outside], img[outside]):
            d = np.abs(res[outside].astype(int) - img[outside].astype(int)).max()
            fail("output differs from input outside mask (max %d)" % d)
        ok("byte-identical outside mask (%d px changed of %d)"
           % (int((mask > 0).sum()), h * w))

        print("[5] watermark removed")
        rg = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        bright_before = (ink).mean()
        bright_after = (rg >= 248).mean()
        if bright_after > 0.002:
            fail("bright residue left in ROI: %.4f of pixels" % bright_after)
        ok("bright pixels: %.4f -> %.4f" % (bright_before, bright_after))
        m = re.search(r"mean abs diff=([\d.]+)", p.stdout)
        if not m or not (0 < float(m.group(1)) < 5):
            fail("unexpected mean abs diff in output: %r" % p.stdout)
        ok("mean abs diff=%s" % m.group(1))

        print("[6] unicode output path")
        uni = os.path.join(tmp, "输出 目录")
        run([src, x0, y0, x1, y1, uni, "--method", "cv2"])
        for suffix in ("_mask.png", "_no_wm.jpg"):
            f = os.path.join(uni, "in" + suffix)
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                fail("unicode path artifact missing: %s" % f)
        ok("wrote to %s" % os.path.basename(uni))

        print("[7] external --mask")
        ext = np.zeros((h, w), np.uint8)
        ext[240:300, 60:200] = 255          # must overlap the ROI, or the mask is empty
        extp = os.path.join(tmp, "ext_mask.png")
        cv2.imwrite(extp, ext)
        out2 = os.path.join(tmp, "out2")
        run([src, x0, y0, x1, y1, out2, "--mask", extp, "--dilate", "0",
             "--method", "cv2"])
        mask2 = cv2.imread(os.path.join(out2, "in_mask.png"), cv2.IMREAD_GRAYSCALE)
        if not np.array_equal(mask2, ext):
            fail("external mask not honoured verbatim")
        res2 = cv2.imread(os.path.join(out2, "in_no_wm.png"), cv2.IMREAD_COLOR)
        keep = mask2 == 0
        if not np.array_equal(res2[keep], img[keep]):
            fail("external-mask run changed pixels outside the mask")
        ok("external mask honoured, %d px rewritten" % int((mask2 > 0).sum()))

        print("[8] empty ROI fails loudly")
        run([src, 0, 0, 8, 8, os.path.join(tmp, "nope"), "--method", "cv2"],
            expect_code=1)
        ok("exits non-zero on empty mask")

        print("[9] SKILL.md frontmatter")
        skill = open(os.path.join(ROOT, "SKILL.md"), encoding="utf-8").read()
        fm = re.match(r"^---\n(.*?)\n---\n", skill, re.S)
        if not fm:
            fail("SKILL.md has no YAML frontmatter")
        body = fm.group(1)
        name = re.search(r"^name:\s*(\S+)", body, re.M)
        desc = re.search(r'^description:\s*"?([^"\n]+)"?', body, re.M)
        if not name or name.group(1) != "image-inpainting":
            fail("frontmatter name must be image-inpainting")
        if not desc:
            fail("frontmatter description missing")
        if len(desc.group(1)) > 60:
            fail("description longer than 60 chars (%d)" % len(desc.group(1)))
        if "license:" not in body:
            fail("frontmatter license missing")
        ok("name=%s, description=%d chars, license present"
           % (name.group(1), len(desc.group(1))))

        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
