"""ACT-lite — 관측 o_t ↦ chunk A_t (k 스텝) + 보조 헤드 Q̂ · F̂.

``docs/POLICY_LEARNING_MATH.md`` §3 의 구조를 작은 규모로 구현한다.

    관측 토큰      프레임 m 장 (CNN 풀링 토큰 + 마지막 프레임 공간 토큰 4×4)
                   + 프레임별 지각 특징 s_t (토큰에 합성)
                   + 벡터 토큰 (Ã_{t−1}, 중력, wrench, 포화신호)
    인코더         TransformerEncoder (무효 프레임은 key padding mask). **z 는 들어가지 않는다.**
    디코더         k 개 학습 쿼리 **+ z_proj(z)** → TransformerDecoder → 스텝별 특징
    헤드
      action       스텝별 (v_x, v_y, ω_z)  [mm/s, mm/s, deg/s]  →  P̂ = Δt·cumsum
      discrete     (선택) 축별 순변위 bin 로짓 (B, 3, n_bins)  §5.3(g)
      quality      Q̂(o, A) — 관측 요약 + 행동 chunk 를 받아 미래 Q k 개 (§3.3)
      force        스텝별 F̂_n (텔레오퍼레이션 데이터에서만 감독)
    CVAE 인코더    q_φ(z | A, o): 라벨 chunk + 벡터 + 마지막 프레임 특징 → (μ, logσ²)

`z = 0` 관례를 쓰지 않는다 (§3.2, L6·L7): 실행시 z 를 M 개 뽑아 Q̂ 로 고른다 (select_action).

**z 주입 지점 (2026-09-08 개정, §3.3-1).** 초판은 z 를 인코더 토큰으로 넣었다. 그러면 self-attention 이
z 를 모든 토큰에 섞어 Q̂ 의 관측 요약 `memory_pooled` 에 라벨(사후분포 z ← A_lab) 이 새어 들어간다 —
학습시 Q̂ 는 라벨을 보고, 실행시엔 후보별 사전분포 z 를 보는 학습·실행 불일치다. 지금은 z 가 디코더
쿼리에만 더해진다. 결과: (1) Q̂ 는 구조적으로 관측만 본다, (2) `select_action` 에서 인코더가 M 회가 아니라
**1 회** 돌고 디코더(쿼리 k 개)만 M 회 돈다 (§8-8 지연 예산).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TimingConfig
from .dataset import OBS_VEC_DIM
from .perception import STATE_DIM

ACTION_DIM = 3
# 행동 정규화 스케일 (Q̂ 입력·CVAE 입력용). mm, mm, deg
ACTION_SCALE = (20.0, 20.0, 10.0)


# --------------------------------------------------------------------------- 블록
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FrameEncoder(nn.Module):
    """(B*m,1,h,w) → 풀링 특징 (B*m, C) 와 공간 특징 맵 (B*m, C, h', w')."""

    def __init__(self, channels: list[int]):
        super().__init__()
        layers = []
        cin = 1
        for c in channels:
            layers.append(ConvBlock(cin, c))
            cin = c
        self.blocks = nn.Sequential(*layers)
        self.out_channels = cin

    def forward(self, x):
        fmap = self.blocks(x)
        return fmap.mean(dim=(2, 3)), fmap


class MLP(nn.Module):
    def __init__(self, din: int, dhid: int, dout: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dhid), nn.GELU(), nn.Dropout(dropout), nn.Linear(dhid, dout))

    def forward(self, x):
        return self.net(x)


def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float()[:, None]
    i = torch.arange(0, d, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------- 모델
@dataclass
class PolicyOutput:
    a_hat: torch.Tensor                 # (B,k,3) 스텝 속도 [mm/s, mm/s, deg/s]
    P_hat: torch.Tensor                 # (B,k+1,3) 누적 변위, P_hat[:,0]=0
    memory_pooled: torch.Tensor         # (B,d) 관측 요약 (Q̂ 입력) — z 를 포함하지 않는다
    F_hat: torch.Tensor                 # (B,k)
    logits: Optional[torch.Tensor]      # (B,3,n_bins) 이산 헤드
    mu: Optional[torch.Tensor]          # (B,z)
    logvar: Optional[torch.Tensor]
    z: Optional[torch.Tensor]


class ActPolicy(nn.Module):
    def __init__(self, mcfg: ModelConfig, timing: TimingConfig):
        super().__init__()
        self.mcfg = mcfg
        self.k = timing.chunk_steps
        self.m = timing.obs_frames
        self.dt = timing.chunk_dt
        d = mcfg.d_model
        self.head_type = mcfg.head
        if self.head_type not in ("cvae", "discrete"):
            raise ValueError(f"model.head 는 cvae|discrete: {self.head_type!r}")
        self.frame_input_size = tuple(int(v) for v in mcfg.frame_input_size)

        # 프레임 인코더 + 토큰화
        self.frame_encoder = FrameEncoder(list(mcfg.frame_channels))
        C = self.frame_encoder.out_channels
        self.frame_proj = nn.Linear(C + STATE_DIM, d)
        self.spatial_proj = nn.Linear(C, d)
        self.spatial_pool = 4
        self.spatial_pos = nn.Parameter(torch.randn(self.spatial_pool ** 2, d) * 0.02)
        self.frame_index_emb = nn.Parameter(torch.randn(self.m, d) * 0.02)
        self.frame_dt_emb = MLP(1, 64, d)
        self.vec_proj = MLP(OBS_VEC_DIM, 128, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)   # vec, (예약), frame, spatial

        # CVAE
        self.z_dim = mcfg.z_dim
        if self.head_type == "cvae":
            self.z_proj = nn.Linear(self.z_dim, d)       # z → 디코더 쿼리 (인코더에는 넣지 않는다)
            self.enc_action_proj = nn.Linear(ACTION_DIM, d)
            self.enc_ctx_proj = nn.Linear(OBS_VEC_DIM + STATE_DIM, d)
            self.enc_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.register_buffer("enc_pos", sinusoid(self.k + 2, d), persistent=False)
            enc_layer = nn.TransformerEncoderLayer(d, mcfg.n_heads, mcfg.dim_feedforward, mcfg.dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
            self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
            self.latent_head = nn.Linear(d, 2 * self.z_dim)

        # 트랜스포머
        enc_layer = nn.TransformerEncoderLayer(d, mcfg.n_heads, mcfg.dim_feedforward, mcfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=mcfg.encoder_layers, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, mcfg.n_heads, mcfg.dim_feedforward, mcfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=mcfg.decoder_layers)
        self.queries = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.enc_norm = nn.LayerNorm(d)
        self.dec_norm = nn.LayerNorm(d)

        # 헤드
        self.action_head = nn.Linear(d, ACTION_DIM)
        self.force_head = nn.Linear(d, 1)
        self.n_bins = mcfg.discrete_bins
        if self.head_type == "discrete":
            self.discrete_head = MLP(d, d, ACTION_DIM * self.n_bins)
        self.quality_head = MLP(d + self.k * ACTION_DIM, mcfg.q_head_hidden, self.k, mcfg.dropout)
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE), persistent=False)

    # ------------------------------------------------------------------ 이산 bin
    def bin_centers(self, device=None) -> torch.Tensor:
        """(3, n_bins) 순변위 bin 중심 [mm, mm, deg]."""
        r = torch.tensor([self.mcfg.discrete_range_mm, self.mcfg.discrete_range_mm,
                          self.mcfg.discrete_range_deg], device=device)
        grid = torch.linspace(-1.0, 1.0, self.n_bins, device=device)
        return r[:, None] * grid[None, :]

    # ------------------------------------------------------------------ 관측 인코딩
    def encode_observation(self, batch: dict[str, torch.Tensor]):
        """관측 → (memory, pad, pooled). z 와 무관하므로 실행시 tick 당 1 회만 부른다."""
        frames = batch["frames"]                       # (B,m,H,W) float [0,1]
        B, m, H, W = frames.shape
        x = frames.reshape(B * m, 1, H, W)
        if (H, W) != self.frame_input_size:
            x = F.interpolate(x, size=self.frame_input_size, mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.25
        pooled, fmap = self.frame_encoder(x)           # (B*m,C), (B*m,C,h,w)
        state = batch["state"].reshape(B * m, -1)
        frame_tok = self.frame_proj(torch.cat([pooled, state], dim=1)).reshape(B, m, -1)
        frame_tok = frame_tok + self.frame_index_emb[None] + self.frame_dt_emb(batch["frame_dt"][..., None]) \
            + self.type_emb[2]

        # 마지막 유효 프레임의 공간 토큰
        valid = batch["frame_valid"]                   # (B,m) bool
        last_idx = (valid.float() * torch.arange(m, device=valid.device)[None]).argmax(dim=1)
        fmap = fmap.reshape(B, m, *fmap.shape[1:])
        last_fmap = fmap[torch.arange(B, device=valid.device), last_idx]      # (B,C,h,w)
        sp = F.adaptive_avg_pool2d(last_fmap, self.spatial_pool).flatten(2).transpose(1, 2)  # (B,16,C)
        sp_tok = self.spatial_proj(sp) + self.spatial_pos[None] + self.type_emb[3]

        vec_tok = (self.vec_proj(batch["vec"]) + self.type_emb[0])[:, None]
        tokens = [vec_tok, frame_tok, sp_tok]
        mask = [torch.zeros(B, 1, dtype=torch.bool, device=valid.device), ~valid,
                torch.zeros(B, sp_tok.shape[1], dtype=torch.bool, device=valid.device)]
        src = torch.cat(tokens, dim=1)
        pad = torch.cat(mask, dim=1)
        memory = self.enc_norm(self.encoder(src, src_key_padding_mask=pad))
        keep = (~pad).float()[..., None]
        pooled_mem = (memory * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return memory, pad, pooled_mem

    # ------------------------------------------------------------------ CVAE 인코더
    def encode_latent(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """q_φ(z | A, o). A 는 라벨 chunk 의 스텝 증분 (정규화)."""
        P = batch["P"]                                             # (B,k+1,3)
        steps = (P[:, 1:] - P[:, :-1]) / self.action_scale          # (B,k,3)
        B = P.shape[0]
        last_state = batch["state"][:, -1]
        ctx = self.enc_ctx_proj(torch.cat([batch["vec"], last_state], dim=1))[:, None]
        seq = torch.cat([self.enc_cls.expand(B, -1, -1), ctx, self.enc_action_proj(steps)], dim=1)
        seq = seq + self.enc_pos[None, : seq.shape[1]]
        h = self.cvae_encoder(seq)[:, 0]
        mu, logvar = self.latent_head(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    # ------------------------------------------------------------------ 디코딩
    def decode(self, memory, pad, B, z: Optional[torch.Tensor] = None):
        q = self.queries[None].expand(B, -1, -1)
        if z is not None:
            q = q + self.z_proj(z)[:, None, :]                                     # z 는 여기서만
        h = self.dec_norm(self.decoder(q, memory, memory_key_padding_mask=pad))   # (B,k,d)
        a_hat = self.action_head(h) * (self.action_scale / (self.k * self.dt))     # 스케일: chunk 전체 ≈ 1 단위
        P_hat = torch.cat([torch.zeros_like(a_hat[:, :1]), torch.cumsum(a_hat, dim=1) * self.dt], dim=1)
        F_hat = self.force_head(h).squeeze(-1)
        logits = None
        if self.head_type == "discrete":
            logits = self.discrete_head(h.mean(1)).reshape(B, ACTION_DIM, self.n_bins)
        return a_hat, P_hat, F_hat, logits

    def predict_quality(self, memory_pooled: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Q̂(o, A). P: (B,k+1,3) 누적 변위 → 미래 Q (B,k)."""
        flat = (P[:, 1:] / self.action_scale).flatten(1)
        return self.quality_head(torch.cat([memory_pooled, flat], dim=1))

    def forward(self, batch: dict[str, torch.Tensor], z: Optional[torch.Tensor] = None,
                use_posterior: bool = True) -> PolicyOutput:
        B = batch["frames"].shape[0]
        mu = logvar = None
        if self.head_type == "cvae":
            if z is None:
                if use_posterior and "P" in batch:
                    mu, logvar = self.encode_latent(batch)
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                else:
                    z = torch.randn(B, self.z_dim, device=batch["frames"].device)
        memory, pad, pooled = self.encode_observation(batch)
        a_hat, P_hat, F_hat, logits = self.decode(memory, pad, B, z)
        return PolicyOutput(a_hat=a_hat, P_hat=P_hat, memory_pooled=pooled, F_hat=F_hat,
                            logits=logits, mu=mu, logvar=logvar, z=z)

    # ------------------------------------------------------------------ 실행시 선택 (§3.3 · §3.5)
    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], n_samples: int = 32, gamma: float = 0.2,
                      prev_dy: Optional[torch.Tensor] = None) -> dict[str, Any]:
        """z 후보 M 개 → Q̂ + 모드 일관성 보너스 (3.3) 로 선택. 이산 헤드는 argmax."""
        B = batch["frames"].shape[0]
        dev = batch["frames"].device
        if self.head_type == "discrete":
            out = self.forward(batch, use_posterior=False)
            centers = self.bin_centers(dev)                                    # (3,n)
            prob = out.logits.softmax(-1)
            net = (prob * centers[None]).sum(-1)                               # 기대값 (B,3)
            idx = out.logits.argmax(-1)
            net_argmax = torch.gather(centers[None].expand(B, -1, -1), 2, idx[..., None]).squeeze(-1)
            ramp = torch.linspace(0, 1, self.k + 1, device=dev)[None, :, None]
            P = net_argmax[:, None, :] * ramp
            Q = self.predict_quality(out.memory_pooled, P)
            margin = (prob.max(-1).values - prob.topk(2, dim=-1).values[..., 1])
            return {"P": P, "a": (P[:, 1:] - P[:, :-1]) / self.dt, "Q_hat": Q, "net_expected": net,
                    "net": net_argmax, "mode_margin": margin, "prob": prob}

        # CVAE: 인코더는 z 와 무관 → 1 회. 디코더만 M 회 (memory 를 M 배로 늘린다)
        memory, pad, pooled = self.encode_observation(batch)
        mem_rep = memory.repeat_interleave(n_samples, dim=0)
        pad_rep = pad.repeat_interleave(n_samples, dim=0)
        pooled_rep = pooled.repeat_interleave(n_samples, dim=0)
        z = torch.randn(B * n_samples, self.z_dim, device=dev)
        _, P_hat, _, _ = self.decode(mem_rep, pad_rep, B * n_samples, z)
        Q = self.predict_quality(pooled_rep, P_hat)                           # (B*M,k)
        score = Q.sum(1)
        P = P_hat.reshape(B, n_samples, self.k + 1, 3)
        Qm = Q.reshape(B, n_samples, self.k)
        score = score.reshape(B, n_samples)
        if prev_dy is not None and gamma > 0:
            sign_match = torch.sign(P[:, :, -1, 1]) * torch.sign(prev_dy)[:, None]
            score = score + gamma * sign_match * self.k * 0.5
        best = score.argmax(1)
        ar = torch.arange(B, device=dev)
        P_best = P[ar, best]
        # 모드 마진: 선택 모드와 반대 부호 v_y 후보 중 최고 점수의 차 (§6 진단 2)
        opp = torch.sign(P[:, :, -1, 1]) != torch.sign(P_best[:, -1, 1])[:, None]
        opp_score = torch.where(opp, score, torch.full_like(score, -1e9)).max(1).values
        margin = score[ar, best] - opp_score
        margin = torch.where(opp.any(1), margin, torch.full_like(margin, float("nan")))
        return {"P": P_best, "a": (P_best[:, 1:] - P_best[:, :-1]) / self.dt, "Q_hat": Qm[ar, best],
                "net": P_best[:, -1], "mode_margin": margin, "candidates": P, "scores": score,
                "vy_spread": P[:, :, -1, 1].std(1)}


def build_policy(mcfg: ModelConfig, timing: TimingConfig) -> ActPolicy:
    return ActPolicy(mcfg, timing)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
