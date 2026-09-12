"""오르막 탐색기 — 측정한 Q 로 방향을 고른다.

학습된 Q̂ 로 고르지 않는 이유는 rus_policy/search.py 머리말에 있다 (방향 판별 52~54 %).
여기서는 로봇 없이, 가상의 Q 지형 위에서 탐색기가 실제로 봉우리를 찾는지 본다.
"""

import numpy as np
import pytest

from rus_policy.search import DWELL, MOVE, SETTLE, UNDO, HillClimbSearch


def _run(peak, n_ticks=3000, hz=5.0, start=(0.0, 0.0), **kw):
    """탐색기를 가상 지형 위에서 돌린다. 각속도를 적분해 자세를 만들고 그 자리의 Q 를 준다."""
    s = HillClimbSearch(**kw)
    ang = np.array(start, float)                      # (θx, θy) [°]
    traj, t = [], 0.0
    for _ in range(n_ticks):
        q = peak(ang)
        w = s.update(t, q)
        ang = ang + np.array(w[:2]) / hz
        traj.append((t, ang.copy(), q, s.phase))
        t += 1.0 / hz
    return s, traj


def _gauss(center=(6.0, -4.0), width=8.0, floor=0.07, top=0.97):
    c = np.asarray(center, float)

    def q(ang):
        d = np.linalg.norm(np.asarray(ang, float) - c)
        return floor + (top - floor) * float(np.exp(-(d / width) ** 2))
    return q


def test_finds_the_peak_from_a_blind_start():
    """방광이 안 보이는 자리에서 출발해 봉우리로 간다."""
    peak = _gauss()
    s, traj = _run(peak, start=(-10.0, 10.0))
    q_start, q_end = traj[0][2], traj[-1][2]
    assert q_end > q_start + 0.3, (q_start, q_end)
    assert q_end > 0.9, q_end
    end = traj[-1][1]
    assert np.linalg.norm(end - np.array([6.0, -4.0])) < 4.0, end


def test_steps_back_when_quality_drops():
    """나빠지면 **한 걸음 되돌아간다** — 직전이 곧 최고점이다."""
    peak = _gauss(center=(0.0, 0.0), width=4.0)
    s, traj = _run(peak, start=(0.0, 0.0), n_ticks=200)
    phases = [p for _, _, _, p in traj]
    assert UNDO in phases, "봉우리에 있는데 되돌림이 한 번도 없었다"
    # 되돌림 뒤에는 출발 자세 근처로 돌아와 있어야 한다
    after = [a for _, a, _, p in traj if p == MOVE]
    assert max(np.linalg.norm(a) for a in after) < 4.0


def test_tries_every_direction_then_dwells():
    """평평한 곳에서는 네 방향을 다 시험하고 쉰다 — 헛돌지 않는다."""
    s, traj = _run(lambda ang: 0.5, n_ticks=400)
    assert DWELL in [p for _, _, _, p in traj], "모든 방향이 실패했는데 대기로 가지 않았다"
    assert s.n_fail <= len(s.axes)


def test_noise_below_min_gain_is_not_taken_as_improvement():
    """Q 잡음(실측 0.0004 수준)으로 방향을 바꾸면 안 된다."""
    rng = np.random.default_rng(0)
    s, traj = _run(lambda ang: 0.5 + rng.normal(0, 0.0005), n_ticks=400, min_gain=0.01)
    assert DWELL in [p for _, _, _, p in traj], "잡음을 개선으로 읽어 계속 전진했다"


def test_blocked_direction_is_undone_and_abandoned():
    """안전 한계에 걸리면 간 만큼 되돌리고 다음 방향으로."""
    s = HillClimbSearch()
    t = 0.0
    for _ in range(2):                                 # 조금 전진
        s.update(t, 0.5); t += 0.2
    first = s.idx
    w = s.update(t, 0.5, blocked=True)
    assert s.phase == UNDO and s.last_decision == "막힘"
    axis, direction = s.axes[first]
    assert np.sign(w[axis]) == -np.sign(direction), w
    for _ in range(40):                                # 되돌림·재측정을 마치면 방향이 바뀐다
        t += 0.2
        s.update(t, 0.5)
    assert s.idx != first


def test_rebases_the_best_after_stepping_back():
    """팬텀이 변하면 옛 최고값은 그 자리의 값이 아니다 — 되돌아온 자리에서 다시 잰다."""
    s = HillClimbSearch(settle_s=0.4, step_deg=2.0, rate_deg_s=3.0)
    t, q = 0.0, 0.9
    while s.phase != UNDO:                             # 한 걸음 갔다가 실패하게 만든다
        s.update(t, q); t += 0.1
        q = 0.5 if s.phase == SETTLE else q            # 나빠졌다
    assert s.best_q == pytest.approx(0.9, abs=1e-6)
    while s.phase in (UNDO,) or s.phase == "재측정":    # 되돌아와 다시 잰다
        s.update(t, 0.4); t += 0.1
        if t > 20: break
    assert s.best_q == pytest.approx(0.4, abs=1e-6), "되돌아온 자리의 값으로 기준을 다시 잡지 않았다"


def test_long_axis_rotation_is_included_by_default():
    """장축 회전(θz)은 기본에 들어 있어야 한다.

    제어 스택은 θz 를 그대로 실행한다 (gate_axes 가 회전 세 축을 통과시키고, 접촉 상한
    0.2 rad/s = 11.5 °/s 는 3 °/s 보다 훨씬 크다). 못 내보내는 것은 정책의 디코더뿐이다
    (후보 산포 0.46° 대 시연 6.29°). 처음에 뺐던 근거("영상면 방향만 바꾼다")는 틀렸다 —
    빔축 둘레로 돌리면 보는 단면 자체가 바뀐다.
    """
    from rus_policy.search import AXES_XY, AXES_XYZ

    s = HillClimbSearch()
    assert {a for a, _ in s.axes} == {0, 1, 2}, s.axes
    assert set(AXES_XYZ) - set(AXES_XY) == {(2, +1.0), (2, -1.0)}
    # 평평한 지형에서는 모든 방향을 훑으므로 θz 차례가 실제로 온다
    seen, t, s2 = set(), 0.0, HillClimbSearch()
    for _ in range(1200):
        for i, v in enumerate(s2.update(t, 0.5)):
            if abs(v) > 1e-9:
                seen.add(i)
        t += 0.1
    assert 2 in seen, f"θz 지령이 한 번도 안 나갔다: {seen}"


def test_describe_is_readable():
    s = HillClimbSearch()
    s.update(0.0, 0.6)
    text = s.describe()
    assert "θx" in text and "최고 Q" in text


def test_runner_uses_measured_quality_not_the_learned_head():
    """탐색 조건은 **잰** Q(q_view)로 방향을 고르고, 지령은 실제로 나가야 한다."""
    from pathlib import Path

    from rus_policy.episode import CONDITIONS

    assert "search" in CONDITIONS
    run = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    blk = run[run.index('if self.condition == "search":'):]
    blk = blk[:blk.index('if self.condition == "placebo"')]
    assert "self.search.update(t, self.q_view" in blk, "잰 Q 가 아니라 다른 값을 쓰고 있다"
    assert "blocked=far" in blk and "_excursion_deg()" in blk, "안전 한계(시작 자세에서의 각)가 없다"
    # hold·expert 로 막히는 분기에 search 가 들어가면 안 된다 (지령이 나가야 한다)
    hold = run[run.index('if self.condition in ("hold", "expert")'):]
    assert '"search"' not in hold[:hold.index("else:")]
    # 에피소드가 열릴 때 탐색기를 다시 시작한다 — 앞 에피소드의 최고값을 들고 가면 안 된다
    assert "self.search.reset(t, self.q_view)" in run

    exp = (Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py").read_text()
    assert '("policy", "placebo", "search")' in exp, "드라이버가 search 에 --execute 를 안 준다"


def test_frames_can_be_saved_so_episodes_become_training_data():
    """오늘 돌린 에피소드가 내일의 학습 데이터가 되려면 영상이 남아야 한다.

    2026-09-12 까지 --save-frames 는 도크스트링에만 있고 구현이 없었다. 로봇이 스스로 방향을
    정해 움직이고 결과를 잰 기록은 조작자가 고른 움직임과 달리 방향의 효과가 뒤섞이지 않아,
    Q̂ 에게 방향을 가르칠 수 있는 유일한 데이터다.
    """
    from pathlib import Path

    run = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert '"--save-frames"' in run, "도크스트링이 약속한 옵션이 없다"
    assert '"--max-saved-frames"' in run, "--loop 은 끝이 없으므로 상한이 필요하다"
    assert 'out / "frames.npz"' in run
    # 관측을 다시 만들려면 프레임과 **그 시각**이 함께 있어야 한다
    save = run[run.index('if self.args.save_frames and self.frames:'):]
    save = save[:save.index("if self.states:")]
    assert "t=np.asarray(self.frame_t" in save and "frames=np.stack" in save
    # 관측용 링버퍼(20 장)와 따로 쌓아야 한다
    assert "self.frames: list = []" in run and "self.buf.append" in run


def test_excursion_cap_stops_every_commanding_condition():
    """시작 자세에서 멀어지면 **어느 조건이든** 지령을 멈춘다.

    정책은 한 방향으로 계속 갈 수 있다 (2026-09-12 실측: 후보는 θx 양수 51 % 로 고른데 Q̂ 로
    고르면 32 %). 90 s × 3 °/s 면 270° 까지 가고, 조작자가 손으로 멈춰야 했다.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert src.count("too_far = self._excursion_deg()") == 1, "한 군데에서만 판정해야 한다"
    assert 'if self.condition in ("hold", "expert") or too_far:' in src, \
        "탐색 조건에만 걸면 정책이 계속 간다"
    # 멈춰도 기록은 이어져야 한다 — 얼마나 갔는지가 곧 그 조건의 결과다
    stop = src[src.index('if self.condition in ("hold", "expert") or too_far:'):]
    assert "self.rows.append" in stop, "멈춘 tick 이 기록에서 빠지면 안 된다"
