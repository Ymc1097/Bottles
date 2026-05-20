#!/usr/bin/env python3
"""
Load an image from a path or HTTP(S) URL and decode QR codes in it.
Requires numpy and opencv-python. For curved labels install pyzbar (ZBar backend).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

import numpy as np

try:
    import cv2
except ImportError as e:
    print("Missing OpenCV: pip install opencv-python numpy", file=sys.stderr)
    raise SystemExit(1) from e


def load_image(path_or_url: str) -> np.ndarray | None:
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(
            path_or_url,
            headers={"User-Agent": "Mozilla/5.0 QRDecoder/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            buf = resp.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.imread(path_or_url, cv2.IMREAD_COLOR)


def decode_qrs_opencv(img: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
    """OpenCV finder + decoder (fast, but often fails on curved labels / glare)."""
    det = cv2.QRCodeDetector()
    texts: list[tuple[str, np.ndarray | None]] = []

    def collect_from(ok: bool, infos, pts_list) -> None:
        if not ok or infos is None:
            return
        pl = pts_list if pts_list is not None else []
        for i, text in enumerate(infos):
            p = pl[i] if i < len(pl) else None
            if isinstance(text, str) and text:
                texts.append((text, p))

    # Multi + single + curved pipelines (different heuristics)
    ok, infos, pts, _ = det.detectAndDecodeMulti(img)
    collect_from(ok, infos, pts)

    curved = det.detectAndDecodeCurved(img)
    if isinstance(curved, tuple) and len(curved) >= 2:
        data, pts = curved[0], curved[1]
        collect_from(True, [data], [pts])

    dec = det.detectAndDecode(img)
    if isinstance(dec, tuple):
        data = dec[0]
        pts = dec[1] if len(dec) > 1 else None
        collect_from(True, [data], [pts] if pts is not None and pts.size else [])

    # Dedupe by text
    seen: set[str] = set()
    out: list[tuple[str, np.ndarray | None]] = []
    for t, p in texts:
        if t not in seen:
            seen.add(t)
            out.append((t, p))
    return out


def decode_qrs_pyzbar(img: np.ndarray) -> list[str]:
    """ZBar-based decode; more robust for rotation, perspective, and mild cylinder warp."""
    try:
        from pyzbar import pyzbar
    except ImportError:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out: list[str] = []
    for sym in pyzbar.decode(rgb):
        try:
            out.append(sym.data.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(sym.data.decode("latin-1"))
    return out


def decode_qrs(img: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
    """Try OpenCV, then ZBar (pyzbar) if available."""
    oc = decode_qrs_opencv(img)
    if oc:
        return oc
    return [(t, None) for t in decode_qrs_pyzbar(img)]


def main() -> None:
    p = argparse.ArgumentParser(description="Decode QR codes from an image path or URL")
    p.add_argument("source", help="Local file path or https:// URL to image")
    args = p.parse_args()

    img = load_image(args.source)
    if img is None:
        print("Could not load image.", file=sys.stderr)
        sys.exit(2)

    results = decode_qrs(img)
    if not results:
        print(
            "No QR code decoded. Common causes: strong glare, heavy blur, or severe "
            "curvature — try filling the frame with the code, diffuse light, straight-on angle."
        )
        try:
            __import__("pyzbar")
        except ImportError:
            print(
                "This Python environment has no pyzbar: run `pip install pyzbar` here "
                "(needs system lib libzbar0 on Debian/Ubuntu). Then retry.",
                file=sys.stderr,
            )
        else:
            print(
                "pyzbar is installed but decoded nothing — try cropping to the QR, "
                "or install libzbar0 if decoding fails silently.",
                file=sys.stderr,
            )
        sys.exit(0)

    for i, (text, _) in enumerate(results, start=1):
        print(f"--- QR #{i} ---")
        print(text)


if __name__ == "__main__":
    main()
