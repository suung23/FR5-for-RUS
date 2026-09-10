"""한글 폰트 선택 — GUI 와 오프라인 그림이 같은 것을 쓰도록 한 곳에 둔다.

GUI 모듈에서 가져다 쓰면 import 만으로 대화형 백엔드가 잡혀 (`matplotlib.use`)
화면 없는 환경의 Agg 렌더링이 깨진다. 그래서 폰트만 따로 뗀다.
"""

from __future__ import annotations


def use_korean_font() -> None:
    """한글이 깨지지 않게 있는 폰트를 고른다. 없으면 조용히 기본값을 쓴다.

    상태줄·수치는 자리를 지켜야 읽히므로 **고정폭**을 쓰는데, 기본 고정폭
    (DejaVu Sans Mono) 에는 한글 자모가 없어 그대로 두면 두부(□) 가 된다.
    그래서 본문용과 고정폭용을 따로 고른다.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "UnDotum"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    mono = [n for n in ("Noto Sans Mono CJK KR", "NanumGothicCoding", "D2Coding") if n in have]
    if mono:
        plt.rcParams["font.monospace"] = mono + list(plt.rcParams["font.monospace"])
    plt.rcParams["axes.unicode_minus"] = False
