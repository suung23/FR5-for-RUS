"""DESIGN_NOTES §3 — 제어 로직 전체 도식 생성기 (표시폭 정렬 보장).

노드 그래프가 아니라 **제어 계층**을 그린다. 띠 하나가 루프 하나이고, 위에서 아래로
느린 루프 → 빠른 루프다. 각 띠 안에 그 층이 실제로 계산하는 법칙을 적는다.
"""
import unicodedata

def w(s):
    return sum(0 if unicodedata.combining(c) else
               (2 if unicodedata.east_asian_width(c) in "WF" else 1) for c in s)

class Canvas:
    def __init__(self, cols, rows):
        self.g = [[" "] * cols for _ in range(rows)]
        self.cols, self.rows = cols, rows
    def put(self, y, x, s):
        for ch in s:
            if not (0 <= y < self.rows):
                continue
            if unicodedata.combining(ch):
                if 0 <= x - 1 < self.cols:
                    self.g[y][x - 1] += ch
                continue
            if 0 <= x < self.cols:
                self.g[y][x] = ch
                if w(ch) == 2 and x + 1 < self.cols:
                    self.g[y][x + 1] = ""
                    x += 1
            x += 1
    def band(self, y, x, width, title, lines):
        """상단 테두리에 제목을 박은 띠."""
        t = "─ " + title + " "
        self.put(y, x, "┌" + t + "─" * (width - 2 - w(t)) + "┐")
        for i, ln in enumerate(lines):
            pad = width - 4 - w(ln)
            assert pad >= 0, "띠 넘침(%d칸): %r" % (-pad, ln)
            self.put(y + 1 + i, x, "│ " + ln + " " * pad + " │")
        self.put(y + 1 + len(lines), x, "└" + "─" * (width - 2) + "┘")
        return y + 2 + len(lines)
    def vtext(self, y, x, s):
        """세로쓰기. 레일 라벨이 띠와 겹치는 것을 피한다."""
        for i, ch in enumerate(s):
            self.put(y + i, x, ch)
    def vline(self, y0, y1, x, ch="│"):
        for y in range(y0, y1 + 1): self.put(y, x, ch)
    def hline(self, y, x0, x1, ch="─"):
        for x in range(x0, x1 + 1): self.put(y, x, ch)
    def render(self):
        out = ["".join(c for c in r if c != "").rstrip() for r in self.g]
        while out and not out[-1]: out.pop()
        return "\n".join(out)

W, X, BW = 116, 4, 104
CX = X + BW // 2
LR, RR = 1, 111                      # wrench 레일 / V_achieved 레일

C = Canvas(W, 130)
y = 0

POS = {}
_band = C.band
def band(y, x, bw, title, lines, key=None):
    y2 = _band(y, x, bw, title, lines)
    if key: POS[key] = (y, y2 - 2)      # (상단, 하단) 행
    return y2
C.band = lambda y, x, bw, title, lines: band(y, x, bw, title, lines)

def arrow(y, label):
    C.vline(y, y, CX); C.put(y, CX + 2, label); C.put(y + 1, CX, "▼")
    return y + 2

y = band(y, X, BW, "오프라인 · 학습   —   US 영상 피드백 + 프로브 궤적", [
  "  US 영상 스트림                          프로브 궤적",
  "  256×256 @ 8 fps  (⏳ §11)               IMU 200 Hz   또는   로봇 FK 125 Hz",
  "        │                                       │",
  "        └───────────────┬───────────────────────┘   pc_unix 단일 시계로 조인",
  "                        ▼                           ⚠ US 고정 지연 미보정 (§14.3)",
  "  분절   [정지 0.3 s → 이동 1~1.5 s → 정지 0.3 s]   ZUPT 괄호 하나 = chunk 하나",
  "                        │        ↳ 연속 궤적에서는 per-step 속도 라벨을 만들 수 없다",
  "        ┌───────────────┴───────────────────────┐      (바이어스가 1 s 에 30 mm/s)",
  "        ▼                                       ▼",
  "  관측  o_t                                라벨  A_t",
  "    원시 16장 + ControlState + Q_seg         순변위 P_k        고가중",
  "    + wrench + roll·pitch                    궤적 형상 R_i     저가중",
  "    + Ṽ_{t−1} 실현운동 + F̄_n* + 포화        σ:  FK 0.1 mm  /  IMU 0.7·τ^1.5 mm",
  "        └───────────────┬───────────────────────┘   ↳ 소스별 σ 가 혼합비를 정한다",
  "                        ▼                              (프리핸드 가중 ≈ FK 의 1/200)",
  "  ACT CVAE  +  Q̂ 헤드  +  F̂_n 헤드          chunk k = 8 (1.6 s @ 5 Hz)",
  "  loss = 이종분산 Huber(순변위·형상) + λ_Q Q̂ + λ_F F̂ + λ_c 실행가능 + λ_r 위험 + β KL",
  "  ⚠ β 를 키우면 posterior collapse → ±d 두 모드가 합쳐진다.  🟡 β = 0.02",
  "  ⚠ 선결 — 시간 동기 · US 프레임률 · Q_seg 타당성(§14.2) · elevational 스윕 부재",
], key='offline')
y = arrow(y, "θ  학습 가중치")

y = band(y, X, BW, "센싱   30 Hz   ⏳ 수집기 실측 8 fps", [
  "us_frame_node  /us/image",
  "  quality_raw_node   고전 영상처리 · 세그 비의존        → Q_raw",
  "  perception_node    Slim U-Net → ControlState          → Q_seg · center_error",
], key='sens')
y = arrow(y, "Q_raw · Q_seg · ControlState")

y = band(y, X, BW, "힘 setpoint   0.2 Hz   force_search_node", [
  "Stage 1a 조대 → 1b 미세    Q̄_raw(F) 구간평균.  최대 품질을 내는 최소 힘",
  "배경 적응   F_n* ± 0.25 N 디더 → 경사 방향으로 0.1 N 이동",
  "2순위 = 최소 위험   C = w₋·max(0, F_n*−F_n)² + w₊·max(0, F_n−F_n*)² ,  w₋/w₊ = 5",
  "  ↳ 접촉 상실 쪽 여유가 얇다.  '최소 힘'을 곧이곧대로 쓰면 매번 얇은 쪽으로 간다 (§5.3)",
], key='force')
y = arrow(y, "F_n*      (policy 관측에는 디더 제거한 F̄_n*)")

y = band(y, X, BW, "상태머신   20 Hz   supervisor", [
  "SEARCH → STAGE1A → STAGE1B → TRACK  ⇄  RECOVER",
  "valid_for_control = False / rejection_reason  →  복구 동작 매핑 (§10.2)",
  "Q̄_raw 25% 열화가 3 s 지속 → STAGE1B 재진입      twist 추종오차 임계 초과 → 정지",
], key='sup')
y = arrow(y, "mode · 영상축 게이트")

y = band(y, X, BW, "영상축   ~5 Hz   policy_node  +  offset_filter", [
  "o_t = ( I_{t−15..t} 원시 16장 , ControlState , wrench , roll·pitch , F̄_n* , Ṽ_{t−1} , 포화 )",
  "ACT     z 후보 32 → Q̂ 최대인 모드 선택 → chunk k=8 → 모드 일관 temporal ensemble",
  "filter  p(d, φ) 격자 ← Q 관측 + Ṽ 실현운동.   ±d 이봉을 명시적으로 유지      [제안]",
  "⚠ 원시 프레임 필수 — 부호 단서가 세그 마스크에 없다.  ROI를 방광으로 crop 금지 (§3.8)",
  "⚠ Q̂ 는 물리 디더를 없애지 못한다.  작고 드물게 만들 뿐 (L8·L11)",
], key='pol')
y = arrow(y, "V_i* = (v_x, v_y, ω_z)      ZOH 20×")

y = band(y, X, BW, "힘축 + 합성   100 Hz   admittance_node  +  QP arbiter", [
  "v_z = clamp( (F_n* − F_n)/B_z , ±10 mm/s )                     B_z = 3000 N·s/m",
  "ω_x = clamp( −M_x/B_r , ±0.2 rad/s )    ω_y = clamp( −M_y/B_r , ±0.2 )   B_r = 0.5",
  "",
  "QP   min ‖W_f S_f(Jq̇ − V_f*)‖² + ‖W_i S_i(Jq̇ − V_i*)‖² + w_r‖q̇‖² + w_m‖q̇ − q̇⁻‖²",
  "     s.t.  관절속도 한계 · ‖J_lin q̇‖ ≤ v_max · 힘 배리어 v_z ≤ α(F_max − F_n)/k_c",
  "     w_f : w_i = 300:1 (무차원화 후) · w_r = w_f λ²  ← DLS λ=0.02 와 등가",
  "     ↳ 역기구학을 흡수한다.  DLS 는 제약 없는 QP 의 특수해 (§9.1)",
], key='qp')
y = arrow(y, "q̇")

y = band(y, X, BW, "서보   125 Hz   servo_node", [
  "q += q̇·dt → ServoJ      워치독 2층 0.1 s      정지 시 지령 선행분 되감기 (§12.3)",
], key='servo')
y = arrow(y, "")

y = band(y, X, BW, "하드웨어", [
  "FR5 오른팔  +  프로브  +  PX6D F/T        피부 접촉 = 정상 상태",
], key='hw')
bottom = y - 1

# 되먹임 레일 두 개 — 실제 층 사이만 잇는다
qt, qb = POS["qp"]; pt, _ = POS["pol"]; st, sb = POS["servo"]

# wrench : servo → admittance/QP  (왼쪽)
C.vline(qt + 1, sb, LR); C.put(qt + 1, LR, "┌"); C.put(sb, LR, "└")
C.hline(qt + 1, LR + 1, X - 1); C.put(qt + 1, X - 1, "►")
C.hline(sb, LR + 1, X - 1); C.put(sb, X - 1, "┘")
C.vtext(qt + 5, LR + 2, "wrench")

# V_achieved : QP → policy/filter  (오른쪽)
C.vline(pt + 1, qb, RR); C.put(pt + 1, RR, "┐"); C.put(qb, RR, "┘")
C.hline(pt + 1, X + BW, RR - 1); C.put(pt + 1, X + BW, "◄")
C.hline(qb, X + BW, RR - 1)
C.vtext(pt + 2, RR - 3, "V_achieved")

print(C.render())
