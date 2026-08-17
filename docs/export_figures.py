#!/usr/bin/env python3
"""docs/architecture.html 의 인라인 SVG를 독립 이미지 파일로 내보낸다.

    python docs/export_figures.py

각 그림을 docs/figures/ 에 .svg 로 뽑고, headless Chrome 이 있으면 2배율 .png 도 만든다.
architecture.html 을 고친 뒤 이 스크립트를 다시 돌리면 그림이 갱신된다.

SVG 는 페이지 스타일시트의 클래스 규칙에 의존하므로, 독립 파일로 성립시키려면
그 규칙과 색 토큰을 파일 안에 직접 심어야 한다. 사본을 두지 않고 페이지의 <style>
블록을 그대로 추출해 심는다 — 사본은 반드시 어긋난다.

PNG 는 항상 라이트 팔레트로 뽑는다. 인쇄본은 뷰어 OS 테마와 무관해야 한다.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "architecture.html")
OUT = os.path.join(HERE, "figures")

#: 문서에 등장하는 순서대로. SVG 개수와 일치해야 한다.
NAMES = [
    "01-architecture",
    "02-axis-split",
    "03-dof-delta",
    "04-force-search",
    "05-state-machine",
    "06-timescale",
    "07-quality-definitions",
    "08-slim-unet",
    "09-slim-vs-standard",
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

def page_stylesheet(html: str) -> str:
    """페이지의 첫 <style> 블록을 그대로 돌려준다.

    사본을 두지 않는 이유: 팔레트나 클래스 규칙을 페이지에서 고쳤는데 여기 사본을
    깜빡하면 문서와 내보낸 그림의 모양이 조용히 갈라진다. SVG 에 안 쓰이는 규칙이
    섞여 들어가도 무해하다 — 일치하는 요소가 없으면 그냥 무시된다.
    """
    match = re.search(r"<style>(.*?)</style>", html, re.S)
    if match is None:
        raise ValueError("architecture.html 에서 <style> 블록을 찾지 못했다")
    # 독립 SVG 에서는 :root 가 <svg> 자신이다. 페이지에서는 `figure svg` 가 주던
    # 전경색을 여기서 직접 준다 — 없으면 currentColor 가 검정으로 떨어진다.
    return match.group(1) + "\n  :root { color: var(--ink-2); }\n"


def find_chrome() -> str | None:
    """PATH 또는 관례적 설치 경로에서 Chromium 계열 실행 파일을 찾는다."""
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def split_svgs(html: str) -> list[str]:
    """문서에 등장한 순서대로 인라인 SVG 조각을 돌려준다."""
    return re.findall(r"<svg\s[^>]*viewBox=[^>]*>.*?</svg>", html, re.S)


def standalone(svg: str, style: str) -> tuple[str, float, float]:
    """인라인 SVG 조각을 네임스페이스와 스타일을 갖춘 독립 문서로 만든다."""
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not box:
        raise ValueError("viewBox 를 찾을 수 없다")
    width, height = float(box.group(1)), float(box.group(2))

    cut = svg.index(">") + 1
    open_tag, body = svg[:cut], svg[cut:]
    if "xmlns" not in open_tag:
        open_tag = open_tag[:-1] + ' xmlns="http://www.w3.org/2000/svg">'
    return open_tag + "\n<style>" + style + "</style>" + body, width, height


def rasterize(chrome: str, svg_doc: str, width: float, height: float, png: str) -> bool:
    """headless Chrome 으로 2배율 PNG 를 만든다. 성공 여부를 돌려준다.

    SVG 를 직접 열면 뷰어가 여백을 넣으므로, 크기를 못 박은 HTML 래퍼를 거친다.
    배경을 명시하지 않으면 투명 PNG 가 나온다.
    """
    # data-theme="light" 를 못 박는다. 인쇄용 PNG 는 뷰어 OS 테마와 무관하게
    # 항상 라이트 팔레트여야 한다.
    wrapper = (
        '<!doctype html><html data-theme="light"><head><meta charset="utf-8"><style>'
        "html,body{margin:0;padding:0;background:#FFFFFF}"
        f"svg{{display:block;width:{width}px;height:{height}px}}"
        "</style></head><body>" + svg_doc + "</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "figure.html")
        io.open(page, "w", encoding="utf-8").write(wrapper)
        try:
            subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    f"--window-size={int(width)},{int(height)}",
                    "--virtual-time-budget=3000",
                    f"--screenshot={png}",
                    "file:///" + page.replace("\\", "/"),
                ],
                check=False,
                capture_output=True,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"    래스터화 실패: {exc}")
            return False
    return os.path.isfile(png)


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"원본이 없다: {SRC}")
        return 1
    os.makedirs(OUT, exist_ok=True)

    html = io.open(SRC, encoding="utf-8").read()
    style = page_stylesheet(html)
    svgs = split_svgs(html)
    if len(svgs) != len(NAMES):
        print(
            f"SVG {len(svgs)}개를 찾았는데 이름은 {len(NAMES)}개다. "
            f"그림을 추가했다면 NAMES 를 갱신하라."
        )
        return 1

    chrome = find_chrome()
    if chrome is None:
        print("Chromium 계열을 찾지 못했다. SVG 만 내보낸다.\n")

    for name, svg in zip(NAMES, svgs):
        doc, width, height = standalone(svg, style)
        svg_path = os.path.join(OUT, name + ".svg")
        io.open(svg_path, "w", encoding="utf-8").write(
            '<?xml version="1.0" encoding="UTF-8"?>\n' + doc
        )

        line = f"{name:20} {int(width):>4} x {int(height):<4}  svg"
        if chrome is not None:
            png_path = os.path.join(OUT, name + ".png")
            if rasterize(chrome, doc, width, height, png_path):
                line += f"  png {int(width) * 2} x {int(height) * 2}"
        print(line)

    print(f"\n{len(NAMES)}개 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
