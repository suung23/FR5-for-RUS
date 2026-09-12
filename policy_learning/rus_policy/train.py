"""학습 루프 + 진단.

진단 (§5.3g · §6 "조용한 실패"):
  * mode_collapse_vy_std  검증 관측에서 z 를 M 개 뽑아 v_y 순변위의 표준편차. 0 으로 가면 붕괴.
  * leak_<축>_sigma       (select_mae − mae) / std(라벨_축) — 사전분포 경로와 사후분포 경로의 오차 차이를
                          축별 **라벨 퍼짐**으로 정규화한 값. 1 에 가까우면 그 축은 관측→행동 정보가
                          **z 로만** 흘렀다는 뜻이다 (인코더가 라벨을 흘려보내고 디코더가 관측을 무시).
                          라벨 불확도(batch sigma_net, 회전 0.1°)로 정규화하면 안 된다 — 목표 1.0 이
                          "사전분포 경로가 사후분포 경로를 0.1° 안에서 따라잡아라" 가 되어 도달할 수 없고,
                          제어기가 어느 축을 보든 β 를 상한까지 밀기만 한다 (2026-09-12).
  * leak_worst_sigma      train.leak_axes 가 고른 축 집합(기본 회전 3 축)의 최악값. β 제어기 기준.
                          β 선택 규칙: leak_worst ≤ beta_adapt_target_sigma 이면서 vy_std > 0 인 최소 β.
  * leak_gap_mm           옛 규약 (y 병진 하나, mm). 호환을 위해 계속 기록만 한다 — **기준으로 쓰지 않는다.**
                          2026-09-12 실측에서 y 만 0.35σ 인데 θy 0.84σ · x 0.85σ 였다. y 는 가속도
                          이중적분 라벨이라 흘려보낼 신호가 적어 구조적으로 작게 나온다.
  * KL 워밍업             β 를 train.beta_warmup_epochs 동안 0 → loss.beta_kl 로 올린다.
  * mode_margin           Q̂ 로 고른 모드와 반대 부호 모드의 점수 차. 잡음 수준이면 대칭 붕괴 (L11).
  * vy_sign_acc           |Δy| > σ 인 샘플에서 v_y 부호 정확도.
  * mae_*                 축별 순변위 절대오차. 6 자유도 전부 (x,y,z mm · θx,θy,θz deg).

  * select_nmae           실행시 경로(사전분포 z + Q̂ 선택)의 σ 정규화 절대오차. 사후분포 경로의 mae_*
                          와 달리 라벨을 보지 않으므로 누설에 면역이다. 체크포인트 기준의 기본값.

체크포인트: <output_dir>/last.pt, best.pt (train.checkpoint_metric 기준, 기본 select_nmae). 이력:
metrics.jsonl. β 는 train.beta_adapt 가 켜져 있으면 위 선택 규칙에 따라 epoch 마다 조정된다.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PolicyConfig, save_config
from .dataset import PolicyH5Dataset
from .losses import compute_loss
from .model import AXIS_NAMES, AXIS_UNITS, Y_AXIS, ActPolicy, build_policy, count_parameters
from .perception import STATE_FEATURE_NAMES

logger = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loaders(cfg: PolicyConfig, dataset_path: Optional[Path] = None
                 ) -> tuple[DataLoader, Optional[DataLoader], PolicyH5Dataset, Optional[PolicyH5Dataset]]:
    path = cfg.resolve(cfg.paths.dataset) if dataset_path is None else Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"데이터셋이 없습니다: {path} — 먼저 scripts/build_dataset.py 를 실행하십시오")
    tr = PolicyH5Dataset(path, split="train", augment=cfg.train.augment,
                         max_samples=cfg.train.max_train_samples, seed=cfg.train.seed)
    va = PolicyH5Dataset(path, split="val", augment=False, seed=cfg.train.seed)
    if len(tr) == 0:
        raise ValueError(f"{path}: train 샘플이 0 개입니다 (split 배정을 확인하십시오)")
    kw = dict(num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=cfg.train.num_workers > 0)
    tl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=len(tr) > cfg.train.batch_size, **kw)
    vl = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, **kw) if len(va) else None
    return tl, vl, tr, va if len(va) else None


#: 누설을 어느 축에서 볼지. 병진은 라벨이 이중적분 오차 지배라 기준으로 못 쓴다 (config 참조).
LEAK_AXIS_SETS = {"rot": ("thx", "thy", "thz"), "all": AXIS_NAMES, "y": ("y",)}


#: 방광이 이만큼은 치우쳐야 "어느 쪽" 이 뜻을 가진다 (2026-09-12 분석과 같은 문턱).
DX_OFFSET_MIN = 0.05
DX_FEATURE = STATE_FEATURE_NAMES.index("centroid_dx")
HAS_MASK_FEATURE = STATE_FEATURE_NAMES.index("has_mask")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """순위상관. 표본이 모자라거나 한쪽이 상수면 NaN."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _masked_mean(ok: torch.Tensor, mask: torch.Tensor) -> float:
    """``mask`` 가 참인 곳의 평균. 하나도 없으면 NaN (그 배치는 집계에서 빠진다)."""
    return ok[mask].float().mean().item() if bool(mask.any()) else float("nan")


def _worst_leak(res: dict[str, float], which: str) -> Optional[tuple[str, float]]:
    """``leak_<축>_sigma`` 중 가장 큰 것 → (축 이름, 값). 후보가 없으면 ``None``."""
    got = [(nm, res[f"leak_{nm}_sigma"])
           for nm in LEAK_AXIS_SETS.get(which, LEAK_AXIS_SETS["rot"])
           if np.isfinite(res.get(f"leak_{nm}_sigma", np.nan))]
    return max(got, key=lambda kv: kv[1]) if got else None


class Trainer:
    def __init__(self, cfg: PolicyConfig, dataset_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.cfg = cfg
        self.device = pick_device(cfg.train.device)
        seed_everything(cfg.train.seed)
        self.out = cfg.resolve(cfg.paths.output_dir) if output_dir is None else Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out / "config.yaml")
        self.train_loader, self.val_loader, self.train_ds, self.val_ds = make_loaders(cfg, dataset_path)
        self.model: ActPolicy = build_policy(cfg.model, cfg.timing).to(self.device)
        logger.info("모델 파라미터 %.2fM, device=%s, train=%d val=%d", count_parameters(self.model) / 1e6,
                    self.device, len(self.train_ds), 0 if self.val_ds is None else len(self.val_ds))
        if cfg.train.init_from:
            ck = torch.load(cfg.resolve(cfg.train.init_from), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model_state"])
            logger.info("가중치만 가져옴: %s (옵티마이저·epoch 는 새로 시작)", cfg.train.init_from)
        self.frozen: list[torch.nn.Module] = []
        if cfg.train.train_only:
            self._freeze_except(cfg.train.train_only)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        steps_per_epoch = max(1, len(self.train_loader))
        total = cfg.train.epochs * steps_per_epoch
        warm = cfg.train.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.use_amp = bool(cfg.train.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.epoch = 0
        self.best = float("inf")
        self.beta_mult = 1.0          # β 자동 조정 배율 (실제 β = loss.beta_kl × 워밍업 × 이 값)
        self.history: list[dict[str, Any]] = []

    #: train_only 값 → 학습할 최상위 모듈.
    TRAINABLE = {"quality": ("quality_head", "quality_base")}

    def _freeze_except(self, which: str) -> None:
        """``which`` 의 모듈만 학습하고 나머지는 얼린다 (config.TrainConfig.train_only)."""
        if which not in self.TRAINABLE:
            raise ValueError(f"train.train_only 는 {sorted(self.TRAINABLE)} 중 하나여야 한다: {which!r}")
        keep = self.TRAINABLE[which]
        n_train = n_all = 0
        for name, p in self.model.named_parameters():
            p.requires_grad = name.split(".")[0] in keep
            n_all += p.numel()
            n_train += p.numel() if p.requires_grad else 0
        self.frozen = [m for n, m in self.model.named_children() if n not in keep]
        # 인코더가 얼었으니 KL 도 β 조정도 의미가 없다. 조정기가 돌면 로그만 어지럽힌다.
        if self.cfg.train.beta_adapt:
            logger.info("train_only=%s — beta_adapt 를 끈다 (인코더가 얼어 있다)", which)
            self.cfg.train.beta_adapt = False
        # 행동 헤드가 얼면 select_nmae 는 상수다 — 그걸로 best 를 고르면 첫 epoch 가 늘 best 다.
        if which == "quality" and self.cfg.train.checkpoint_metric == "select_nmae":
            logger.info("train_only=quality — checkpoint_metric 을 select_nmae → qual 로 바꾼다")
            self.cfg.train.checkpoint_metric = "qual"
        logger.info("학습 %s: %d / %d 파라미터 (%.1f %%) · 얼린 모듈 %d 개는 eval 모드로 묶는다",
                    which, n_train, n_all, 100.0 * n_train / max(n_all, 1), len(self.frozen))

    def _pin_frozen_eval(self) -> None:
        """얼린 모듈을 eval 로 되돌린다. model.train() 이 전부 train 으로 바꾸므로 매번 부른다."""
        for m in self.frozen:
            m.eval()

    # ------------------------------------------------------------------ 체크포인트
    def save(self, name: str) -> Path:
        path = self.out / name
        torch.save({
            "model_state": self.model.state_dict(), "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.sched.state_dict(), "scaler_state": self.scaler.state_dict(),
            "epoch": self.epoch, "best": self.best, "config": self.cfg.to_dict(),
            "best_metric": self.cfg.train.checkpoint_metric, "beta_mult": self.beta_mult,
            "state_features": self.train_ds.state_features,
        }, path)
        return path

    def resume(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        self.sched.load_state_dict(ck["scheduler_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self.epoch = int(ck["epoch"])
        self.best = float(ck["best"])
        self.beta_mult = float(ck.get("beta_mult", 1.0))
        prev_metric = ck.get("best_metric")
        cur_metric = self.cfg.train.checkpoint_metric
        if prev_metric is not None and prev_metric != cur_metric:
            logger.warning("체크포인트 기준이 %s → %s 로 바뀌어 best 를 초기화합니다 (이전 값 %.4f 는 비교 불가)",
                           prev_metric, cur_metric, self.best)
            self.best = float("inf")
        logger.info("재개: %s (epoch %d, best %.4f, β×%.3f)", path, self.epoch, self.best, self.beta_mult)

    # ------------------------------------------------------------------ 한 epoch
    def beta_scale(self) -> float:
        """KL 워밍업 × 자동 조정 배율. 워밍업: 첫 epoch 1/warm, beta_warmup_epochs 부터 1 (선형)."""
        warm = int(self.cfg.train.beta_warmup_epochs)
        ramp = 1.0 if warm <= 0 else float(min(1.0, (self.epoch + 1) / warm))
        return ramp * self.beta_mult

    def adapt_beta(self, va: dict[str, float]) -> None:
        """헤더의 β 선택 규칙을 epoch 마다 적용한다.

        누설(gap > target)이면 z 의 대역폭을 좁히고, 붕괴(vy_std 가 σ 아래)면 넓히고, 둘 다 여유가
        있으면 "최소 β" 쪽으로 되돌린다. 누설이 배포 성능을 직접 망가뜨리므로 우선순위가 가장 높다.
        """
        t = self.cfg.train
        if not t.beta_adapt or self.cfg.loss.beta_kl <= 0:
            return
        if self.model.head_type != "cvae":       # 잠재변수가 없는 헤드는 KL 항이 0 이라 β 가 무의미
            return
        if self.epoch < max(1, int(t.beta_warmup_epochs)):    # 워밍업 중에는 건드리지 않는다
            return
        sig = va.get("sigma_net_y_mm")
        vy = va.get("mode_collapse_vy_std")
        # 누설은 **σ 로 정규화한 최악 축**으로 본다 (t.leak_axes, 기본 회전 3 축). 옛 규약처럼
        # y 병진 하나만 보면 그 축이 이중적분 라벨이라 누설이 구조적으로 작게 나와, 정작 신호가
        # 있는 회전 축이 통째로 새는 동안 제어기가 β 를 계속 풀어 준다 (2026-09-12).
        worst = _worst_leak(va, t.leak_axes)
        if worst is None:
            return
        leak_axis, gap_sigma = worst
        target = t.beta_adapt_target_sigma
        before = self.beta_mult
        acc = va.get("select_vy_sign_acc")
        selector_ok = acc is not None and np.isfinite(acc) and acc > t.beta_adapt_selector_acc
        if gap_sigma > target:                                        # 누설 — 좁힌다 (최우선)
            self.beta_mult *= t.beta_adapt_rate
        elif not selector_ok:
            pass                    # Q̂ 가 못 고르는 동안은 z 를 넓혀 봐야 실행시 오차만 커진다
        elif vy is not None and np.isfinite(vy) and vy < t.beta_adapt_collapse_sigma * sig:
            self.beta_mult /= t.beta_adapt_rate                       # 붕괴 — 넓힌다
        elif gap_sigma < 0.5 * target:                                # 여유 — 최소 β 쪽으로
            self.beta_mult /= t.beta_adapt_rate
        lo = t.beta_min / self.cfg.loss.beta_kl
        hi = t.beta_max / self.cfg.loss.beta_kl
        self.beta_mult = float(np.clip(self.beta_mult, lo, hi))
        if self.beta_mult != before:
            logger.info("    β %.3f → %.3f  (누설 최악 %s %.2fσ / 목표 %.2fσ, vy_std %s)",
                        self.cfg.loss.beta_kl * before, self.cfg.loss.beta_kl * self.beta_mult,
                        leak_axis, gap_sigma, target, "n/a" if vy is None else f"{vy:.2f}")

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        self._pin_frozen_eval()
        agg: dict[str, list[float]] = {}
        t0 = time.time()
        bscale = self.beta_scale()
        for it, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch, use_posterior=True)
                loss, logs = compute_loss(self.model, out, batch, self.cfg.loss, beta_scale=bscale)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            for k, v in logs.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(v)
            if self.cfg.train.log_every and (it + 1) % self.cfg.train.log_every == 0:
                logger.info("  ep %d it %d/%d loss %.4f", self.epoch + 1, it + 1, len(self.train_loader), logs["total"])
        res = {k: float(np.mean(v)) for k, v in agg.items()}
        res["time_s"] = time.time() - t0
        res["lr"] = float(self.opt.param_groups[0]["lr"])
        return res

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader]) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        agg: dict[str, list[float]] = {}
        spreads, margins, sel_sign = [], [], []
        sel_ae, sel_nae, sig_y, lab_net, sel_dir = [], [], [], [], []
        sel_net, obs_dx, obs_seen = [], [], []
        dir_thr = self._dir_threshold(loader).to(self.device)
        for batch in loader:
            batch = to_device(batch, self.device)
            out = self.model(batch, use_posterior=True)
            _, logs = compute_loss(self.model, out, batch, self.cfg.loss)
            for k, v in logs.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(v)
            # 실행시 선택 경로의 진단 (§6): z 표본 → Q̂ 선택
            sel = self.model.select_action(batch, n_samples=self.cfg.train.z_samples_eval,
                                           gamma=self.cfg.train.gamma_mode_consistency,
                                           prev_dy=batch["vec"][:, 1] * batch["vec"][:, 3])
            P_lab = batch["P"]
            if "vy_spread" in sel:
                spreads.append(sel["vy_spread"].mean().item())
            m = sel["mode_margin"]
            m = m[torch.isfinite(m)]
            if m.numel():
                margins.append(m.mean().item())
            # 실행시 경로는 라벨을 보지 않으므로 누설에 면역인 유일한 정확도 지표다
            sel_ae_b = (sel["net"] - P_lab[:, -1]).abs()
            sel_ae.append(sel_ae_b.mean(0).cpu().numpy())
            sel_nae.append((sel_ae_b / batch["sigma_net"].clamp_min(1e-6)).mean().item())
            sig_y.append(batch["sigma_net"][:, 1].cpu().numpy())
            # 누설은 **라벨 퍼짐**으로 정규화한다 — sigma_net 은 라벨 불확도(회전 0.1°)라
            # 그 단위로 재면 목표 1.0σ 가 "사전분포 경로가 사후분포 경로를 0.1° 안에서 따라잡아라"
            # 가 되어 도달할 수 없고, 제어기는 어느 축을 보든 β 를 상한까지 밀기만 한다.
            # 퍼짐으로 재면 0~1 이 "신호의 몇 할을 잃었나" 가 된다 (2026-09-12).
            lab_net.append(P_lab[:, -1].cpu().numpy())
            big = P_lab[:, -1, Y_AXIS].abs() > batch["sigma_net"][:, Y_AXIS]
            if big.any():
                sel_sign.append((torch.sign(sel["net"][big, Y_AXIS]) == torch.sign(P_lab[big, -1, Y_AXIS])).float().mean().item())
            # 축별 **방향** 정확도. MAE 로는 방향 학습이 안 보인다 — centroid_dx→θy 선형 프로브가
            # test 상관 0.575 를 내면서 MAE 는 2.36→2.37 로 제자리였다 (2026-09-12). 크기는 대부분
            # 0 근처 표본이 정하고, 방향은 크게 움직인 표본에만 있다. 그래서 |라벨| 이 큰 쪽만 본다.
            ok = torch.sign(sel["net"]) == torch.sign(P_lab[:, -1])
            sel_dir.append(np.stack([
                _masked_mean(ok[:, i], P_lab[:, -1, i].abs() > dir_thr[i]) for i in range(len(AXIS_NAMES))]))
            # 관측→행동 규칙을 직접 잰다: 방광이 치우친 쪽과 고른 행동의 상관.
            # 부호 정확도만으로는 안 된다 — 시연자 라벨은 자기상관이 세서(직전 chunk 부호로
            # 찍으면 θy 93 % · θx 83 %) "직전 움직임 이어가기" 만으로 높은 점수가 나온다.
            # 이 상관은 시연자 +0.665 (θy) 가 목표값이고, 지금 모델은 +0.02 다 (2026-09-12).
            sel_net.append(sel["net"].cpu().numpy())
            obs_dx.append(batch["state"][:, -1, DX_FEATURE].cpu().numpy())
            obs_seen.append((batch["state"][:, -1, HAS_MASK_FEATURE] > 0.5).cpu().numpy())
        res = {k: float(np.mean(v)) for k, v in agg.items()}
        if spreads:
            res["mode_collapse_vy_std"] = float(np.mean(spreads))
        if margins:
            res["mode_margin"] = float(np.mean(margins))
        if sel_ae:
            m = np.mean(np.stack(sel_ae), axis=0)
            for i, (nm, un) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
                res[f"select_mae_{nm}_{un}"] = float(m[i])
        if sel_nae:
            res["select_nmae"] = float(np.mean(sel_nae))
        if sig_y:
            res["sigma_net_y_mm"] = float(np.median(np.concatenate(sig_y)))
        if sel_sign:
            res["select_vy_sign_acc"] = float(np.mean(sel_sign))
        if "select_mae_y_mm" in res and "mae_y_mm" in res:
            res["leak_gap_mm"] = res["select_mae_y_mm"] - res["mae_y_mm"]   # 옛 규약 (호환용)
        if lab_net:
            spread = np.concatenate(lab_net).std(axis=0)
            for i, (nm, un) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
                res[f"spread_{nm}_{un}"] = float(spread[i])
                a, c = res.get(f"mae_{nm}_{un}"), res.get(f"select_mae_{nm}_{un}")
                if a is not None and c is not None and spread[i] > 0:
                    res[f"leak_{nm}_sigma"] = float((c - a) / spread[i])
            worst = _worst_leak(res, self.cfg.train.leak_axes)
            if worst is not None:
                res["leak_worst_sigma"] = worst[1]
        if sel_dir:
            d = np.nanmean(np.stack(sel_dir), axis=0)
            for i, nm in enumerate(AXIS_NAMES):
                if np.isfinite(d[i]):
                    res[f"select_dir_{nm}"] = float(d[i])
            axes = LEAK_AXIS_SETS.get(self.cfg.train.leak_axes, LEAK_AXIS_SETS["rot"])
            got = [res[f"select_dir_{nm}"] for nm in axes if f"select_dir_{nm}" in res]
            if got:
                # 낮을수록 좋은 값으로 둔다 — checkpoint_metric 은 최솟값을 고른다.
                res["select_dir_err"] = float(1.0 - np.mean(got))
        if sel_net:
            net, dx = np.concatenate(sel_net), np.concatenate(obs_dx)
            keep = np.concatenate(obs_seen) & (np.abs(dx) > DX_OFFSET_MIN)
            if keep.sum() >= 30:
                for i, nm in enumerate(AXIS_NAMES):
                    res[f"select_corr_dx_{nm}"] = _spearman(dx[keep], net[keep, i])
                res["n_corr"] = float(keep.sum())
                # 면내 규칙을 얼마나 배웠나. 시연자 +0.665 가 목표, 낮을수록 좋게 부호를 뒤집는다.
                res["inplane_err"] = float(1.0 - res["select_corr_dx_thy"])
        return res

    def _dir_threshold(self, loader: DataLoader) -> np.ndarray:
        """방향을 셀 표본을 고르는 문턱 — 축별 라벨 퍼짐. 검증셋마다 한 번만 잰다."""
        if getattr(self, "_dir_thr", None) is None:
            lab = np.concatenate([b["P"][:, -1].numpy() for b in loader])
            self._dir_thr = torch.from_numpy(lab.std(axis=0).astype(np.float32))
        return self._dir_thr

    # ------------------------------------------------------------------ 전체
    def fit(self) -> list[dict[str, Any]]:
        log_path = self.out / "metrics.jsonl"
        while self.epoch < self.cfg.train.epochs:
            tr = self.train_epoch()
            va = self.evaluate(self.val_loader)
            self.epoch += 1
            row = {"epoch": self.epoch, **{f"train/{k}": v for k, v in tr.items()},
                   **{f"val/{k}": v for k, v in va.items()}}
            self.history.append(row)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            metric = self.cfg.train.checkpoint_metric
            score = va.get(metric, va.get("total", tr["total"]))
            msg = (f"epoch {self.epoch}/{self.cfg.train.epochs}  train {tr['total']:.4f}  "
                   f"val {va.get('total', float('nan')):.4f}  "
                   f"mae(x,y,z)=({va.get('mae_x_mm', float('nan')):.2f},{va.get('mae_y_mm', float('nan')):.2f},"
                   f"{va.get('mae_z_mm', float('nan')):.2f})"
                   f" mae(θx,θy,θz)=({va.get('mae_thx_deg', float('nan')):.2f},"
                   f"{va.get('mae_thy_deg', float('nan')):.2f},{va.get('mae_thz_deg', float('nan')):.2f})")
            if "mode_collapse_vy_std" in va:
                msg += f"  vy_std {va['mode_collapse_vy_std']:.2f}"
            if "mode_margin" in va:
                msg += f"  margin {va['mode_margin']:.3f}"
            if "leak_gap_mm" in va:
                msg += f"  leak_gap {va['leak_gap_mm']:.2f}mm"
            if "select_nmae" in va:
                msg += f"  sel_nmae {va['select_nmae']:.3f}"
            msg += f"  beta {tr.get('beta', float('nan')):.3f}"
            # 품질 손실을 늘 찍는다. 헤드만 재학습할 때(train_only=quality)는 행동 헤드가 얼어
            # 위의 total·mae·sel_nmae 가 epoch 마다 같게 나오고, 정작 학습 중인 값은 이것뿐이다 —
            # 2026-09-11 서버 재학습에서 요약 줄만 보고는 헤드가 배우는지 알 수 없었다.
            if "qual" in va:
                msg += f"  qual {tr.get('qual', float('nan')):.4f}/{va['qual']:.4f}"
            logger.info(msg)
            self.adapt_beta(va)          # 저장 전에 — last.pt 는 다음 epoch 에 쓸 β 를 담아야 한다
            self.save("last.pt")
            if score < self.best:
                self.best = score
                self.save("best.pt")
        return self.history


def load_policy(path: str | Path, device: str = "auto") -> tuple[ActPolicy, PolicyConfig]:
    """학습된 체크포인트 → (모델, 설정)."""
    dev = pick_device(device)
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = PolicyConfig.from_dict(ck["config"])
    # 저장된 설정에 없는 키는 from_dict 가 **현재 기본값**으로 채운다. 그래서 기능이 추가되면 옛
    # 체크포인트가 새 구조로 조립돼 state_dict 가 어긋난다. 가중치를 보고 되돌린다.
    if "quality_base.net.0.weight" not in ck["model_state"]:
        cfg.model.quality_residual = False
    model = build_policy(cfg.model, cfg.timing).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg
