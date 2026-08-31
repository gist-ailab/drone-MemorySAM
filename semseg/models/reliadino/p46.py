"""[P46-CTR] Class-Transfer Recovery — P39.1-rank 위 3개 학습-전용 토글 모듈.

설계 정본: `.claude_logs/decisions/2026-07-28-p46-classtransfer-recovery-proposal.md`
진단: DELIVER val→test 붕괴의 지배 원인 = **per-class 도메인 전이 붕괴**
(Wall 62→2, TrafficLight 81→13, Water 33→0, Bridge 46→0). 두 하위원인 =
(a) rare-class under-learning, (b) 도메인 간 class 표현 붕괴.

이 파일은 세 처방의 **기제**만 담는다(결선은 model.py / train_reliadino.py):

  C-1  Adaptive Rare-Class Sampling      [DAFormer 2111.14887]
       train 셋 per-class 픽셀빈도(1회 캐시) → P(c) ∝ exp((1−f_c)/T)
       → class c 샘플 → c를 포함한 이미지 샘플. 런타임 per-class EMA loss로
       추가 blend(고-loss 클래스 up-weight). **주 CE/M2F 손실이 이 데이터를 본다.**

  C-2  Masked-Context Consistency        [MIC 2212.01322, source-only DG 변형]
       student=패치마스킹 입력, teacher=EMA 복사본·원본 입력. 마스킹 영역에서
       teacher pseudo-label에 consistency. UDA와 달리 **target 도메인 불요** —
       source에 regularizer로 건다(우리 setting 적응).

  C-3  Domain-Invariant Class-Prototype Consistency  [2309.14282 / SCSD 2412.12050]
       per-class EMA prototype bank(K×D) + 픽셀 feature의 prototype-contrastive
       CE(자기 prototype으로 당기고 타 prototype에서 민다). ColorAugSSD 스타일
       2-view 간 같은 클래스 → 같은 prototype = 도메인불변화.

🔴 전 항목 **학습 전용**: 추론(model.eval()) 경로는 P39.1과 완전히 동일하다.
🔴 키1 준수: 세 항목 모두 주손실/aux 손실로 **직접 gradient**를 낸다
   (zero-init 잔차·수동 0-게이트 아님 — 4연속 반증된 실패키).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Sampler

# ─────────────────────────────────────────────────────────────────────────────
# C-1 · Adaptive Rare-Class Sampling
# ─────────────────────────────────────────────────────────────────────────────

# 라벨 파일 경로/디코딩은 데이터셋마다 다르다. **추정하지 않고 명시**한다 —
# 잘못 디코딩한 빈도표는 조용히 틀린 샘플링 분포를 만들고(ISSUE-025류), 실험
# 전체를 오염시킨다. 새 데이터셋을 붙일 때 여기 어댑터를 추가할 것.
#   DELIVER : {root}/semantic/... , 원본 1-25 + 255 → (l[l==255]=0; l-=1) = 0-24 + 255
#   MUSES   : dataset._sibling(rgb, 'mask') , 이미 trainId 0-18 + 255
_LABEL_ADAPTERS = ('DELIVER', 'MUSES')


def _label_path(dataset, idx: int) -> str:
    name = type(dataset).__name__
    f = str(dataset.files[idx])
    if name == 'MUSES':
        return dataset._sibling(f, 'mask')
    if name == 'DELIVER':
        return f.replace('/img', '/semantic').replace('_rgb', '_semantic')
    raise NotImplementedError(
        f"[P46-C1] {name}용 라벨 경로 어댑터가 없다. p46._LABEL_ADAPTERS에 추가하라 "
        f"(현재 지원: {_LABEL_ADAPTERS}).")


def _read_label(dataset, idx: int) -> np.ndarray:
    from torchvision import io
    name = type(dataset).__name__
    lbl = io.read_image(_label_path(dataset, idx))[0]          # (H,W) uint8
    if name == 'DELIVER':
        # deliver.py __getitem__과 **동일한** 변환 (1-25→0-24, 0→255 underflow)
        lbl = lbl.clone()
        lbl[lbl == 255] = 0
        lbl -= 1
    return lbl.numpy()


class _LabelStatDataset(torch.utils.data.Dataset):
    """빈도 집계용 경량 셋 — 라벨 1장만 읽어 bincount를 돌려준다(멀티워커용)."""

    def __init__(self, base, num_classes: int):
        self.base = base
        self.K = num_classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        lbl = _read_label(self.base, i)
        cnt = np.bincount(lbl.reshape(-1), minlength=256)[:self.K]
        return i, torch.from_numpy(cnt.astype(np.int64))


def _stats_cache_key(dataset, num_classes: int) -> str:
    sig = json.dumps({
        'cls': type(dataset).__name__,
        'n': len(dataset),
        'K': num_classes,
        'first': str(dataset.files[0]),
        'last': str(dataset.files[-1]),
    }, sort_keys=True)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def compute_class_stats(dataset, num_classes: int, cache_dir: str,
                        min_pixels: int = 1, num_workers: int = 8,
                        verbose: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """train 셋의 (per-class 픽셀수, 클래스별 보유 이미지 인덱스)를 1회 산출·캐시.

    Returns
      pix   : (K,) int64  클래스별 총 픽셀 수
      files : list of K int64 arrays — 클래스 c를 `min_pixels` 이상 담은 이미지 인덱스
    """
    cache = Path(cache_dir) / f"rcs_{type(dataset).__name__}_{_stats_cache_key(dataset, num_classes)}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        pix = z['pix']
        per = z['per']                                  # (N, K) int64
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        per = np.zeros((len(dataset), num_classes), dtype=np.int64)
        loader = torch.utils.data.DataLoader(
            _LabelStatDataset(dataset, num_classes), batch_size=16,
            num_workers=num_workers, shuffle=False)
        seen = 0
        for idxs, cnts in loader:
            per[idxs.numpy()] = cnts.numpy()
            seen += len(idxs)
            if verbose and seen % 512 < 16:
                print(f"[P46-C1] class-stat scan {seen}/{len(dataset)}", flush=True)
        pix = per.sum(0)
        np.savez_compressed(cache, pix=pix, per=per)
        if verbose:
            print(f"[P46-C1] class stats cached -> {cache}", flush=True)
    files = [np.nonzero(per[:, c] >= min_pixels)[0].astype(np.int64)
             for c in range(num_classes)]
    return pix, files


def rcs_base_prob(pix: np.ndarray, temperature: float = 0.01,
                  mode: str = 'daformer') -> np.ndarray:
    """클래스 샘플 확률.

    mode='daformer' (기본, DAFormer 2111.14887 식 4):
        f_c = pix_c / Σpix ,  P(c) ∝ exp((1 − f_c) / T)  , T=0.01
        T가 작을수록 희소 클래스에 강하게 쏠린다.
    mode='power':
        P(c) ∝ f_c^(−T)  — 제안서 본문의 문자 표기. ⚠️ T=0.01이면 사실상
        균등분포(=no-op)라서 기본값이 아니다. 이 모드를 쓸 거면 T≳0.5로 올려라.
    빈 클래스(pix=0)는 확률 0 — 뽑아도 이미지가 없다.
    """
    pix = pix.astype(np.float64)
    present = pix > 0
    f = pix / max(pix.sum(), 1.0)
    if mode == 'power':
        p = np.where(present, np.power(np.maximum(f, 1e-12), -float(temperature)), 0.0)
    else:
        z = (1.0 - f) / max(float(temperature), 1e-6)
        z = z - z[present].max() if present.any() else z        # overflow 방지
        p = np.where(present, np.exp(z), 0.0)
    s = p.sum()
    return (p / s) if s > 0 else np.full_like(p, 1.0 / len(p))


class ClassLossEMA:
    """런타임 per-class CE 손실의 EMA (C-1 난이도 신호 — 외부 라벨 없이 내부신호).

    학습 루프가 매 스텝 `update()`, 샘플러가 `weights()`를 읽어 base 분포와 blend.
    DDP: rank별 독립 유지(collective 없음). rank마다 자기 shard 통계로 자기
    샘플을 뽑으며 기대분포는 동일 — 2026-07-16 NCCL 데드락 이력을 감안해 매
    스텝 all_reduce를 새로 켜지 않는다.
    """

    def __init__(self, num_classes: int, momentum: float = 0.99):
        self.K = num_classes
        self.m = float(momentum)
        self.val = np.zeros(num_classes, dtype=np.float64)
        self.seen = np.zeros(num_classes, dtype=bool)

    @torch.no_grad()
    def update_from_logits(self, logits: torch.Tensor, target: torch.Tensor,
                           ignore_label: int = 255) -> None:
        """logits (B,K,H,W) / target (B,H,W) → per-class 평균 CE로 EMA 갱신."""
        K = logits.shape[1]
        ce = F.cross_entropy(logits.detach().float(), target, reduction='none',
                             ignore_index=ignore_label)                    # (B,H,W)
        t = target.reshape(-1)
        keep = t != ignore_label
        if not bool(keep.any()):
            return
        t = t[keep]
        c = ce.reshape(-1)[keep]
        s = torch.zeros(K, device=c.device, dtype=torch.float32).scatter_add_(0, t, c)
        n = torch.zeros(K, device=c.device, dtype=torch.float32).scatter_add_(
            0, t, torch.ones_like(c))
        s = s.cpu().numpy().astype(np.float64)
        n = n.cpu().numpy().astype(np.float64)
        hit = n > 0
        mean = np.zeros(K, dtype=np.float64)
        mean[hit] = s[hit] / n[hit]
        # 처음 본 클래스는 EMA warm-up 없이 즉시 반영 (0에서 서서히 오르는 동안
        # rare 클래스가 오히려 down-weight 되는 역전 방지)
        fresh = hit & (~self.seen)
        self.val[fresh] = mean[fresh]
        old = hit & self.seen
        self.val[old] = self.m * self.val[old] + (1.0 - self.m) * mean[old]
        self.seen |= hit

    def weights(self) -> np.ndarray:
        """평균 1로 정규화한 난이도 가중 (미관측 클래스는 1.0 = 중립)."""
        w = np.ones(self.K, dtype=np.float64)
        if self.seen.any():
            v = self.val[self.seen]
            mu = float(v.mean())
            if mu > 1e-8:
                w[self.seen] = v / mu
        return w

    def state_dict(self):
        return {'val': self.val.tolist(), 'seen': self.seen.tolist()}

    def load_state_dict(self, d):
        if not d:
            return
        self.val = np.asarray(d['val'], dtype=np.float64)
        self.seen = np.asarray(d['seen'], dtype=bool)


class RareClassSampler(Sampler):
    """[C-1] class-우선 샘플러 (복원추출). DistributedSampler 대체.

    매 draw: c ~ P(c) → c를 포함한 이미지 중 균등 샘플.
      P(c) ∝ base(c) · (1 + blend_w · ĝ_c),  ĝ = ClassLossEMA.weights()
    blend는 `refresh` draw마다 재계산 → epoch 중에도 난이도 변화를 따라간다.

    DDP: rank마다 독립 시드로 자기 몫(len//world)을 뽑는다. 복원추출이라
    shard 분할(DistributedSampler)과 달리 중복/누락 개념이 없고 collective도
    필요 없다. `set_epoch()`로 epoch마다 스트림이 바뀐다.
    """

    def __init__(self, class_files: List[np.ndarray], base_prob: np.ndarray,
                 num_samples: int, rank: int = 0, world_size: int = 1,
                 seed: int = 0, loss_ema: Optional[ClassLossEMA] = None,
                 blend_w: float = 1.0, refresh: int = 32):
        self.class_files = class_files
        # 이미지가 하나도 없는 클래스는 뽑히면 안 된다(무한 재추첨 방지)
        avail = np.array([len(f) > 0 for f in class_files], dtype=bool)
        p = np.where(avail, base_prob, 0.0)
        s = p.sum()
        if s <= 0:
            raise RuntimeError("[P46-C1] 샘플 가능한 클래스가 없다 — 빈도표를 확인하라")
        self.base_prob = p / s
        self.num_samples = int(num_samples)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed, self.epoch = int(seed), 0
        self.loss_ema = loss_ema
        self.blend_w = float(blend_w)
        self.refresh = max(1, int(refresh))
        self.last_class_hist = np.zeros(len(class_files), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _prob(self) -> np.ndarray:
        p = self.base_prob
        if self.loss_ema is not None and self.blend_w > 0:
            p = p * (1.0 + self.blend_w * self.loss_ema.weights())
            s = p.sum()
            p = p / s if s > 0 else self.base_prob
        return p

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + 1000003 * self.epoch + 7919 * self.rank)
        self.last_class_hist = np.zeros(len(self.class_files), dtype=np.int64)
        p = self._prob()
        for i in range(self.num_samples):
            if i % self.refresh == 0:
                p = self._prob()                     # 런타임 난이도 반영
            c = int(rng.choice(len(p), p=p))
            pool = self.class_files[c]
            self.last_class_hist[c] += 1
            yield int(pool[rng.integers(len(pool))])


# ─────────────────────────────────────────────────────────────────────────────
# C-2 · Masked-Context Consistency (MIC, source-only DG 변형)
# ─────────────────────────────────────────────────────────────────────────────

def patch_mask(b: int, h: int, w: int, ratio: float, patch: int,
               device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """(B,1,H,W) float 마스크 — 1 = **가려진** 패치. MIC 식 랜덤 패치 마스킹."""
    ph, pw = max(1, math.ceil(h / patch)), max(1, math.ceil(w / patch))
    m = (torch.rand(b, 1, ph, pw, device=device, generator=generator) < ratio).float()
    return F.interpolate(m, size=(h, w), mode='nearest')


def apply_patch_mask(inputs: Sequence[torch.Tensor], mask: torch.Tensor,
                     modal_idx: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
    """마스킹된 영역을 0(= Normalize 후 데이터셋 평균)으로 지운다.

    modal_idx=None이면 **전 모달 동일 위치**를 지운다 — 국소 외형 대신 주변
    context로 추론하게 만드는 것이 목적이므로, img만 지우면 lidar/depth가 그
    영역을 그대로 메워 압력이 사라진다.
    """
    keep = 1.0 - mask
    out = list(inputs)
    tgt = range(len(out)) if modal_idx is None else modal_idx
    for i in tgt:
        out[i] = out[i] * keep
    return out


def masked_consistency_loss(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            mask: torch.Tensor,
                            conf_thresh: float = 0.75,
                            mode: str = 'ce') -> Tuple[torch.Tensor, float]:
    """마스킹 영역에서 student ↔ teacher pseudo-label consistency.

    teacher는 **원본(비마스킹)** 입력을 본 EMA 복사본이므로, 이 손실은
    "가려진 영역을 주변 context로 복원하라"는 요구다. 학습 초기에는 teacher가
    미숙 → conf_thresh를 넘는 픽셀이 거의 없어 손실≈0 (자연스러운 warmup).

    Returns (loss, 통과 픽셀 비율).
    """
    with torch.no_grad():
        p = F.softmax(teacher_logits.float(), dim=1)
        conf, pseudo = p.max(1)                                   # (B,H,W)
        sel = (conf >= conf_thresh).float() * mask[:, 0]
        rate = float(sel.mean())
        denom = sel.sum().clamp(min=1.0)
    if mode == 'kl':
        logq = F.log_softmax(student_logits.float(), dim=1)
        with torch.no_grad():
            pt = p
        per = (pt * (torch.log(pt.clamp(min=1e-8)) - logq)).sum(1)   # (B,H,W)
    else:
        per = F.cross_entropy(student_logits.float(), pseudo, reduction='none')
    return (per * sel).sum() / denom, rate


class EMATeacher:
    """student의 EMA 복사본 (학습 전용, optimizer/DDP 밖).

    - **frozen 파라미터는 student와 저장소를 공유**한다: 이 리포의 백본(ViT-L)은
      전부 frozen이라 통째 deepcopy하면 ~1.2GB를 헛되이 쓴다. frozen은 학습
      중 절대 바뀌지 않으므로 공유가 정확히 등가다.
    - teacher는 항상 eval() + no_grad → 학습 대상이 아니고, 추론 경로에도
      존재하지 않는다(체크포인트에도 안 들어간다).
    """

    def __init__(self, model: nn.Module, alpha: float = 0.999):
        self.alpha = float(alpha)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._pairs: List[Tuple[nn.Parameter, nn.Parameter]] = []
        shared = 0
        for (ns, ps), (nt, pt) in zip(model.named_parameters(),
                                      self.ema.named_parameters()):
            assert ns == nt, f"[P46-C2] teacher 파라미터 순서 불일치: {ns} vs {nt}"
            if ps.requires_grad:
                self._pairs.append((ps, pt))
            else:
                pt.data = ps.data                 # frozen → 저장소 공유
                shared += 1
        self.n_shared, self.n_ema = shared, len(self._pairs)

    @torch.no_grad()
    def update(self, step: int) -> None:
        """MIC/DACS 관행: a_t = min(1 − 1/(t+1), alpha) — 초기엔 빠르게 따라붙는다."""
        a = min(1.0 - 1.0 / (step + 1.0), self.alpha)
        for ps, pt in self._pairs:
            pt.data.mul_(a).add_(ps.data.detach(), alpha=1.0 - a)

    @torch.no_grad()
    def __call__(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        self.ema.eval()
        out = self.ema(list(inputs), True)
        logits = out[0]
        del out
        self._clear_diag()
        return logits

    def _clear_diag(self) -> None:
        """🔴 메모리: teacher의 eval-경로 분석 탭(`_last_*`)을 즉시 해제한다.

        ReliaDINO.forward는 `not self.training` 분기에서 분석용 텐서를 모듈에
        캐시한다 — `_last_per_modal_feats`(모달 수 × (B,1024,h,w)),
        `_last_fused_postfusion` / `_last_fused_prehead`, `_last_per_modal_outputs`,
        `_last_p43_out`(M2F 출력 dict: pred_masks (B,Q,H/4,W/4) 포함). 이 탭들은
        val_*/tools/viz_* 가 **student**에서 읽는 것이고 teacher에서는 아무도
        읽지 않는다. 그런데 teacher는 매 스텝 호출되므로, 지워 주지 않으면
        다음 스텝의 student forward 2회가 peak를 찍는 내내 그 캐시가 살아 있다.
        (teacher 전용 — student `_core`의 탭은 건드리지 않는다.)
        """
        for m in self.ema.modules():
            for k, v in list(m.__dict__.items()):
                if k.startswith('_last_') and v is not None:
                    m.__dict__[k] = None

    def set_epoch(self, epoch: int) -> None:
        self.ema._current_epoch = epoch


# ─────────────────────────────────────────────────────────────────────────────
# C-3 · Domain-Invariant Class-Prototype Consistency
# ─────────────────────────────────────────────────────────────────────────────

_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


def style_jitter_normalized(img: torch.Tensor,
                            mean: Sequence[float] = _IMNET_MEAN,
                            std: Sequence[float] = _IMNET_STD,
                            brightness_delta: float = 32.0 / 255.0,
                            contrast: Tuple[float, float] = (0.5, 1.5),
                            saturation: Tuple[float, float] = (0.5, 1.5),
                            hue_delta: float = 0.1) -> torch.Tensor:
    """ColorAugSSD(augmentations_mm.ColorAugSSD)와 **동일 강도**의 photometric
    변주를 이미 Normalize된 배치 텐서에 적용한다 (2번째 style view 생성용).

    데이터로더를 두 번 돌지 않으므로 **기하 변환이 view1과 정확히 일치**한다
    (같은 crop/flip) — prototype 정합에 필요한 픽셀 대응이 보장된다.
    ⚠️ img(카메라) 모달 전용. ColorAugSSD도 비-camera 모달에는 identity다.
    """
    m = torch.tensor(mean, device=img.device, dtype=img.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=img.device, dtype=img.dtype).view(1, -1, 1, 1)
    x = (img * s + m).clamp(0.0, 1.0)                     # → [0,1] RGB
    out = []
    for i in range(x.shape[0]):
        v = x[i:i + 1]
        if torch.rand(()) < 0.5:                          # SSD: 가산 밝기
            v = (v + (torch.rand(()) * 2 - 1).item() * brightness_delta).clamp(0, 1)
        contrast_first = bool(torch.rand(()) < 0.5)
        if contrast_first and torch.rand(()) < 0.5:
            v = TF.adjust_contrast(v, _u(contrast))
        if torch.rand(()) < 0.5:
            v = TF.adjust_saturation(v, _u(saturation))
        if torch.rand(()) < 0.5:
            v = TF.adjust_hue(v, (torch.rand(()) * 2 - 1).item() * hue_delta)
        if (not contrast_first) and torch.rand(()) < 0.5:
            v = TF.adjust_contrast(v, _u(contrast))
        out.append(v)
    x = torch.cat(out, 0).clamp(0.0, 1.0)
    return (x - m) / s


def _u(rng: Tuple[float, float]) -> float:
    return float(rng[0] + torch.rand(()).item() * (rng[1] - rng[0]))


class PrototypeBank(nn.Module):
    """[C-3] per-class EMA prototype bank + prototype-contrastive 손실.

    손실 = CE( cos(f_i, P) / τ , y_i ) — 자기 prototype으로 **당기고** 동시에 타
    prototype에서 **민다**(softmax 분모). prototype은 detach된 EMA라 손실의
    gradient는 전부 **feature 쪽**(head→FPN→fusion→LoRA)으로 흐른다 = 키1.

    bank는 매 스텝 여러 스타일 증강 view의 클래스 평균으로 갱신되므로 그 자체가
    **style-marginalized** 타깃이다. 여기에 2-view(원본 / ColorAugSSD 변주)를
    같은 bank로 당기면 "도메인이 달라도 같은 클래스는 같은 prototype" 제약이 된다.

    DDP 주의: 보조 branch(C-2/C-3 2-view)가 켜지면 train_reliadino.py가
    `broadcast_buffers=False`로 DDP를 만든다(2번째 forward의 버퍼 브로드캐스트가
    1번째 forward 그래프의 버퍼를 in-place로 갈아엎어 backward가 죽기 때문 —
    tools/smoke_p46.py --ddp 로 실측). 그 경우 이 bank는 **rank-로컬 EMA**다.
    rank마다 타깃이 미세하게 다르지만 전부 같은 분포의 iid 표본이고 DDP가
    gradient를 평균하므로 무해하다(sync 없는 BN과 같은 계약).
    """

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.999,
                 temperature: float = 0.1, pixels: int = 4096,
                 ignore_label: int = 255):
        super().__init__()
        self.K, self.D = int(num_classes), int(dim)
        self.m = float(momentum)
        self.tau = float(temperature)
        self.pixels = int(pixels)
        self.ignore_label = int(ignore_label)
        self.register_buffer('proto', torch.zeros(self.K, self.D))
        self.register_buffer('inited', torch.zeros(self.K))
        self._last_cov = 0.0        # 진단: 이번 스텝에 관측된 클래스 수 / K

    def _sample(self, feat: torch.Tensor, gt: torch.Tensor):
        """feat (B,D,h,w) + gt (B,H,W) → 서브샘플된 (P,D) feature / (P,) label.

        🔴 메모리: **인덱싱을 먼저** 하고 fp32 캐스팅은 마지막에 한다.
        이전 구현은 `feat.float()` → `permute().reshape()` → `f[keep]` 순서라
        (B,D,h,w) **전체 크기의 fp32 사본을 3장** autograd 그래프에 남긴 뒤
        `pixels`(4096)행만 썼다 — DELIVER 768²·fpn_dim 256 기준 호출당 ~110MiB,
        보조 branch까지 스텝당 2회. 아래 gather는 (P,D)만 그래프에 올린다.

        뽑히는 행·난수 소비·수치는 이전과 **완전히 동일**하다:
          · `keep` 판정과 `torch.randint(0, n, (pixels,))` 호출이 그대로다(같은 n).
          · flat row r ↔ (b = r//hw, p = r%hw)는 `permute(0,2,3,1).reshape(-1,D)`의
            행 순서와 정확히 같은 대응이다.
          · bf16→fp32 캐스팅은 무손실이라 gather 후 캐스팅해도 값이 같다.
        """
        B, D, h, w = feat.shape
        hw = h * w
        g = F.interpolate(gt.unsqueeze(1).float(), size=(h, w),
                          mode='nearest').squeeze(1).long().reshape(-1)
        keep = (g != self.ignore_label) & (g >= 0) & (g < self.K)
        idx = keep.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return None, None
        if self.pixels > 0 and n > self.pixels:
            sel = torch.randint(0, n, (self.pixels,), device=feat.device)
            idx = idx[sel]
        # (B,D,hw) 뷰에 advanced index → 결과 (P,D). 전체 사본이 생기지 않는다.
        f = feat.reshape(B, D, hw)[idx.div(hw, rounding_mode='floor'),
                                   :, idx % hw].float()
        return f, g[idx]

    @torch.no_grad()
    def _update(self, f: torch.Tensor, g: torch.Tensor) -> None:
        classes = torch.unique(g)
        for c in classes.tolist():
            mu = f[g == c].mean(0).float()
            if float(self.inited[c]) < 0.5:
                self.proto[c] = mu
                self.inited[c] = 1.0
            else:
                self.proto[c].mul_(self.m).add_(mu, alpha=1.0 - self.m)

    def forward(self, feat: torch.Tensor, gt: torch.Tensor,
                update: bool = True,
                class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """prototype-contrastive CE. bf16 autocast 아래에서도 fp32로 계산한다.

        [P52] class_weights (K,) — 클래스별 λ_c. 주면 weighted CE(F.cross_entropy
        표준 semantics: sum(w·ce)/sum(w))로, None이면 기존 균일 평균 그대로다.
        λ_c≈0인 건강 클래스는 당기는 압력이 0이 된다(C3-adaptive — 붕괴 클래스만
        prototype 당김). bank 갱신(_update)은 가중과 무관하게 항상 같은 클래스
        평균으로 한다 — bank가 가중돼면 prototype 자체가 왜곡된다.
        """
        with torch.autocast(device_type=feat.device.type, enabled=False):
            # `.float()`는 _sample이 **서브샘플 후에** 건다 (전체 사본 회피).
            f, g = self._sample(feat, gt)
            if f is None:
                return feat.new_zeros(())
            # 아직 한 번도 안 본 클래스는 이번 배치 평균으로 즉시 초기화해야
            # 분모에 참여할 수 있다 (0-벡터 prototype은 정규화 시 NaN).
            with torch.no_grad():
                cold = torch.unique(g)
                cold = cold[self.inited[cold] < 0.5]
                for c in cold.tolist():
                    self.proto[c] = f[g == c].mean(0).float()
                    self.inited[c] = 1.0
                live = self.inited > 0.5
                self._last_cov = float(live.float().mean())
            fn = F.normalize(f, dim=1)
            pn = F.normalize(self.proto.float(), dim=1).detach()
            logits = (fn @ pn.t()) / max(self.tau, 1e-4)          # (N,K)
            logits = logits.masked_fill(~live.view(1, -1), float('-inf'))
            if class_weights is None:
                loss = F.cross_entropy(logits, g)
            else:
                w = class_weights.detach().to(device=logits.device,
                                             dtype=logits.dtype)
                if float(w.sum()) <= 0.0:
                    # [P52] warmup 등 전 클래스 λ_c=0 → weighted CE의 분모가
                    # 0이 되어 NaN을 뿌린다. 손실은 0, bank 갱신은 계속(EMA
                    # prototype은 λ와 무관하게 학습돼야 warmup 종료 직후 바로
                    # 쓸 수 있다).
                    loss = logits.new_zeros(())
                else:
                    loss = F.cross_entropy(logits, g, weight=w)
            if update:
                self._update(f.detach(), g)
            return loss


# ─────────────────────────────────────────────────────────────────────────────
# 공통 스케줄
# ─────────────────────────────────────────────────────────────────────────────

def ramp(epoch: int, warmup_ep: int) -> float:
    """0→1 선형 ramp (warmup_ep<=0이면 항상 1)."""
    if warmup_ep <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_ep)))
