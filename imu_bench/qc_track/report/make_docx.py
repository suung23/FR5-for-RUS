#!/usr/bin/env python3
"""아티팩트용 HTML -> Word(.docx).

    python3 make_docx.py

LibreOffice 의 HTML 임포터는 CSS 변수도 flex/grid 도 data URI 도 못 읽는다. 그래서
아티팩트 파일을 그대로 넘기면 표는 살아도 레이아웃이 통째로 무너진다. 여기서는
**같은 내용을 Word 가 이해하는 마크업으로 다시 조립한다** — 블록 레이아웃은 표로,
그림은 옆에 둔 파일 참조로, 스타일은 요소 인라인으로.

내용은 body.html 한 곳에서만 읽는다. 두 벌로 갈라지면 하나는 반드시 낡는다.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess

from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
# 그림과 산출물은 qc_track 아래에 있다. 경로를 절대경로로 박으면 다른 사람이 못 쓴다.
QC = os.path.dirname(HERE)
FIGS = os.path.join(QC, "raw_data", "meta", "analysis")
FIG_FILE = {"{{FIG_SOURCES}}": "report_sources.png",
            "{{FIG_MAIN}}": "qc_host6.png",
            "{{FIG_DISP}}": "report_displacement.png"}
FIG_WIDTH = {"report_sources.png": 640, "qc_host6.png": 640,
             "report_displacement.png": 560}

INK, MUTED, RULE = "#111820", "#5d6a78", "#c9d2dc"
ACCENT, GOOD, BAD, WARN = "#17567e", "#276b45", "#a3261d", "#8a5a0c"
SANS = "'맑은 고딕','Malgun Gothic','IBM Plex Sans KR',sans-serif"
MONO = "'D2Coding','Consolas','IBM Plex Mono',monospace"


def tag(soup, name, **attrs):
    return soup.new_tag(name, **attrs)


def cell(soup, text, *, head=False, style=""):
    td = soup.new_tag("th" if head else "td")
    td["style"] = style
    if hasattr(text, "name") or isinstance(text, list):
        for x in (text if isinstance(text, list) else [text]):
            td.append(x)
    else:
        td.string = str(text)
    return td


def build(src, out_html):
    soup = BeautifulSoup(open(src, encoding="utf-8").read(), "lxml")

    for t in soup.find_all(["style", "link"]):
        t.decompose()

    # --- 그림: data URI 토큰 -> 옆에 둔 파일 -------------------------------
    for img in soup.find_all("img"):
        s = img.get("src", "")
        name = FIG_FILE.get(s)
        if name is None:                        # 이미 치환된 파일이면 이름으로 찾는다
            for tok, f in FIG_FILE.items():
                if tok in s:
                    name = f
                    break
        if name is None:
            continue
        # **data URI 로 넣어야 그림이 문서 안에 박힌다.** 파일 경로로 두면
        # LibreOffice 가 링크로만 걸어서 word/media 가 비고 (실제로 그렇게 나왔다),
        # 다른 PC 에서 열면 그림이 통째로 깨진다.
        with open(os.path.join(FIGS, name), "rb") as fh:
            img["src"] = ("data:image/png;base64,"
                          + base64.b64encode(fh.read()).decode())
        img["width"] = str(FIG_WIDTH.get(name, 600))
        for k in ("alt",):
            img[k] = img.get(k, "")

    # --- 사양 레일 (dl.spec) -> 2 열 표 ------------------------------------
    spec = soup.find("dl", class_="spec")
    if spec:
        tb = soup.new_tag("table", border="1", cellspacing="0", cellpadding="5")
        tb["style"] = f"border-collapse:collapse;width:100%;font-size:9.5pt;color:{INK};"
        for div in spec.find_all("div", recursive=False):
            dt, dd = div.find("dt"), div.find("dd")
            tr = soup.new_tag("tr")
            tr.append(cell(soup, dt.get_text(" ", strip=True), head=True,
                           style=f"width:26%;text-align:left;background:#eef2f6;"
                                 f"font-family:{SANS};font-weight:600;"))
            tr.append(cell(soup, list(dd.children),
                           style=f"text-align:left;font-family:{SANS};"))
            tb.append(tr)
        spec.replace_with(tb)

    # --- 머릿수치 (.headline-nums) -> 1 행 표 ------------------------------
    hn = soup.find("div", class_="headline-nums")
    if hn:
        tb = soup.new_tag("table", border="1", cellspacing="0", cellpadding="7")
        tb["style"] = "border-collapse:collapse;width:100%;"
        tr = soup.new_tag("tr")
        for num in hn.find_all("div", class_="num"):
            col = GOOD if "pass" in num.get("class", []) else BAD
            b, sp = num.find("b"), num.find("span")
            inner = soup.new_tag("div")
            big = soup.new_tag("div")
            big["style"] = (f"font-family:{MONO};font-size:17pt;font-weight:bold;"
                            f"color:{col};")
            big.string = b.get_text(strip=True)
            small = soup.new_tag("div")
            small["style"] = f"font-size:8.5pt;color:{MUTED};font-family:{SANS};"
            for x in list(sp.children):
                small.append(x)
            inner.append(big); inner.append(small)
            tr.append(cell(soup, inner, style="text-align:left;vertical-align:top;"))
        tb.append(tr)
        hn.replace_with(tb)

    # --- 상자(.answer/.note) -> 테두리 한 칸 표 ----------------------------
    for cls, edge, bg in (("answer", ACCENT, "#f4f8fb"),
                          ("note", WARN, "#fbf6ea")):
        for box in soup.find_all("div", class_=cls):
            if cls == "note" and "hard" in box.get("class", []):
                edge, bg = BAD, "#fbf0ef"
            elif cls == "note" and "plain" in box.get("class", []):
                edge, bg = RULE, "#f2f5f8"
            tb = soup.new_tag("table", border="0", cellspacing="0", cellpadding="10")
            tb["style"] = "border-collapse:collapse;width:100%;"
            tr = soup.new_tag("tr")
            td = soup.new_tag("td")
            td["style"] = (f"border-left:4px solid {edge};background:{bg};"
                           f"font-family:{SANS};")
            for x in list(box.children):
                td.append(x)
            tr.append(td); tb.append(tr)
            box.replace_with(tb)

    # --- 자료표 -----------------------------------------------------------
    for tw in soup.find_all("div", class_="tw"):
        t = tw.find("table")
        if not t:
            continue
        t["border"] = "1"; t["cellspacing"] = "0"; t["cellpadding"] = "4"
        t["style"] = (f"border-collapse:collapse;width:100%;font-size:9pt;"
                      f"font-family:{SANS};color:{INK};")
        for th in t.find_all("th"):
            th["style"] = (f"background:#eef2f6;font-weight:600;font-size:8.5pt;"
                           f"text-align:{'left' if th.find_previous_sibling() is None else 'right'};")
        for td in t.find_all("td"):
            cl = td.get("class", [])
            st = "text-align:right;"
            if td.find_previous_sibling() is None:
                st = "text-align:left;"
            if "n" in cl:
                st += f"font-family:{MONO};"
            if "good" in cl:
                st += f"color:{GOOD};font-weight:600;"
            if "bad" in cl:
                st += f"color:{BAD};font-weight:600;"
            if "ref" in cl:
                st += f"color:{MUTED};"
            td["style"] = st
            # 숫자 칸은 짧은 값 하나뿐이라 줄바꿈될 자리가 없어야 한다. LibreOffice 가
            # white-space:nowrap 을 표 칸에서 안 지켜 "-21 / ms" 로 쪼개졌다.
            if "n" in cl:
                for txt in list(td.strings):
                    if " " in txt:
                        txt.replace_with(txt.replace(" ", "\u00a0"))

        # 표 설명은 <caption> 으로 두면 Word 가 **표 위에 가운데 정렬**로 올린다.
        # 읽는 순서가 뒤집히므로 표 뒤의 문단으로 내린다.
        cap = t.find("caption")
        cap_p = None
        if cap:
            cap_p = soup.new_tag("p")
            cap_p["style"] = (f"font-family:{SANS};font-size:8pt;color:{MUTED};"
                              f"line-height:1.45;margin:2pt 0 10pt;")
            for x in list(cap.children):
                cap_p.append(x)
            cap.decompose()
        tw.replace_with(t)
        if cap_p is not None:
            t.insert_after(cap_p)

    # --- figure/figcaption -> 문단 두 개 ----------------------------------
    # LibreOffice 가 figure 를 블록으로 다루지 못해 figcaption 안의 <b>그림 N.</b> 이
    # 오른쪽에 따로 떨어져 나갔다. 그림 문단과 설명 문단으로 평평하게 편다.
    for fig in soup.find_all("figure"):
        img = fig.find("img")
        cap = fig.find("figcaption")
        holder = soup.new_tag("div")
        if img is not None:
            pi = soup.new_tag("p")
            pi["style"] = "text-align:center;margin:10pt 0 3pt;"
            pi.append(img.extract())
            holder.append(pi)
        if cap is not None:
            pc = soup.new_tag("p")
            pc["style"] = (f"font-family:{SANS};font-size:8.5pt;color:{MUTED};"
                           f"line-height:1.5;margin:0 0 12pt;")
            for x in list(cap.children):
                pc.append(x)
            holder.append(pc)
        fig.replace_with(holder)
        holder.unwrap()

    # --- 상태 칩 -> 굵은 글씨 ---------------------------------------------
    for ch in soup.find_all("span", class_="chip"):
        cl = ch.get("class", [])
        col = {"p": GOOD, "f": BAD, "w": WARN}.get(
            next((c for c in cl if c in ("p", "f", "w", "n")), "n"), MUTED)
        ch.name = "b"
        ch["style"] = f"color:{col};font-family:{SANS};font-size:8.5pt;"
        del ch["class"]

    # --- 본문 요소 --------------------------------------------------------
    for h in soup.find_all("h1"):
        h["style"] = (f"font-family:{SANS};font-size:20pt;font-weight:700;"
                      f"color:{INK};line-height:1.25;margin:0 0 6pt;")
    for h in soup.find_all("h2"):
        n = h.get("data-n")
        if n and h.string:
            h.string = f"{n}. {h.get_text(strip=True)}"
        h["style"] = (f"font-family:{SANS};font-size:13.5pt;font-weight:700;"
                      f"color:{INK};border-bottom:2px solid {INK};"
                      f"padding-bottom:3pt;margin:18pt 0 8pt;")
    for h in soup.find_all("h3"):
        h["style"] = (f"font-family:{SANS};font-size:10.5pt;font-weight:700;"
                      f"color:{ACCENT};margin:12pt 0 4pt;")
    for p in soup.find_all("p"):
        st = p.get("style", "")
        if "kicker" in (p.get("class") or []):
            continue
        p["style"] = st + f"font-family:{SANS};font-size:10pt;line-height:1.6;color:{INK};"
    for p in soup.find_all("p", class_="standfirst"):
        p["style"] = (f"font-family:{SANS};font-size:11.5pt;line-height:1.55;"
                      f"color:{MUTED};margin:0 0 10pt;")
    for d in soup.find_all(class_="kicker"):
        d["style"] = (f"font-family:{MONO};font-size:8.5pt;color:{ACCENT};"
                      f"letter-spacing:1pt;margin:0 0 4pt;")
    for li in soup.find_all("li"):
        li["style"] = f"font-family:{SANS};font-size:10pt;line-height:1.6;margin-bottom:4pt;"
    for c in soup.find_all("code"):
        c["style"] = f"font-family:{MONO};font-size:9pt;"
    for fc in soup.find_all("figcaption"):
        fc["style"] = f"font-family:{SANS};font-size:8.5pt;color:{MUTED};line-height:1.5;"
    for e in soup.find_all("em"):
        e["style"] = f"font-style:normal;color:{ACCENT};font-weight:600;"
    for f in soup.find_all("footer"):
        f["style"] = (f"border-top:1px solid {RULE};padding-top:8pt;margin-top:20pt;"
                      f"font-family:{MONO};font-size:8pt;color:{MUTED};")

    body_inner = soup.find("div", class_="page")
    html = ["<!doctype html>", '<html lang="ko"><head><meta charset="utf-8">',
            "<title>프로브 IMU 추적 정확도 — QC 결과</title>",
            f'<style>body{{font-family:{SANS};color:{INK};}}'
            f'table{{page-break-inside:avoid;}} img{{max-width:100%;}}</style>',
            "</head><body>", str(body_inner), "</body></html>"]
    open(out_html, "w", encoding="utf-8").write("\n".join(html))
    return out_html


def main() -> int:
    work = os.path.join(HERE, "docx_build")
    os.makedirs(work, exist_ok=True)
    src = os.path.join(HERE, "body.html")
    tmp = os.path.join(work, "qc_report_word.html")
    build(src, tmp)

    # --infilter 를 안 주면 "Writer/Web document" 로 들어가 페이지 설정이 없는
    # 웹 레이아웃 문서가 된다. Word 에서 인쇄하거나 페이지를 나눌 수 없다.
    r = subprocess.run(["soffice", "--headless", "--norestore",
                        "--infilter=HTML (StarWriter)",
                        "--convert-to", "docx:MS Word 2007 XML",
                        "--outdir", work, tmp],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout.strip() or r.stderr.strip())
    out = os.path.join(work, "qc_report_word.docx")
    if not os.path.exists(out):
        print("변환 실패")
        return 1
    import zipfile
    with zipfile.ZipFile(out) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    if len(media) < len(FIG_FILE):
        print(f"  ! 그림이 {len(media)}/{len(FIG_FILE)} 개만 박혔다 — 링크로 걸렸을 수 있다")
        return 1
    print(f"  그림 {len(media)} 개 문서 안에 박힘")
    final = os.path.join(QC, "IMU_추적정확도_QC_20260821.docx")
    shutil.move(out, final)
    print(f"-> {final}  ({os.path.getsize(final) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
