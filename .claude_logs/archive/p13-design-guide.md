---
legacy_file: P13_design_guide.md
moved: 2026-07-08
---

# P13 설계 가이드: Energy-Confidence Fusion + Expert Collapse Fix

> 🗄 **[ARCHIVED — 2026-02 동결]** P13 설계 시점 가이드. 구현 결과/한계는 [02_model_arch.md](02_model_arch.md) P13 섹션, [06_result_analysis_P13.md](06_result_analysis_P13.md) 참조.
> 작성일: 2026-02-25
> 기반 모델: P9 (LoRA_Sam_P9, M-score 81.47)
> 목표: CrossModalFusionHead 상수 수렴 문제 해결 + MoE expert collapse 해결

---

## 1. 동기 및 배경

### P9의 두 가지 문제

**문제 1: CrossModalFusionHead가 상수 함수로 수렴**
- 모든 이미지(val/test 345장)에서 UAMM/AMF 가중치가 동일:
  - UAMM: img=0.745, lidar=0.961, thermal=1.0
  - AMF: img=0.275, lidar=0.355, thermal=0.370
- 원인: GAP + LayerNorm 정규화 → 입력 무관하게 같은 벡터 → 같은 출력
- 결과: 밤에 RGB가 쓸모없어도 27.5% 가중치를 계속 부여

**문제 2: MoE Expert Collapse (Block9 기준 E1 사망)**
- Block 6-20 (15개 블럭, 전체의 62.5%)에서 E1 사용률 < 3%
- 원인: experts_b zero-init → 대칭 시작 → rich-get-richer
- 결과: 3-expert MoE가 실질적으로 2-expert로 동작, 용량 1/3 낭비

### P10/P11이 실패한 이유 (반복하지 말 것)

- P10: GT oracle KL loss → 학습/test 메커니즘 불일치 → test 과적합 (M: 79.27)
- P11: MI routing loss → 이미 정상인 gate에 불필요한 제약 → 악화 (M: 77.09)
- **교훈**: learned gating은 상수로 수렴하기 쉬움. computed signal을 써야 함.

---

## 2. P13 아키텍처 개요

```
P9에서 변경되는 부분:
[교체] CrossModalFusionHead → ConfidenceAuxHead (공유 1개) + Energy-based weight 계산
[수정] SoftMoE_LoRA_Layer.reset_parameters() → experts_b 비영 초기화
[유지] 나머지 전부 (backbone, memory attention, max-norm UAMM, AMF, 학습 파이프라인)
```

### Forward 흐름

```
Phase 1: 모달리티별 인코딩 (P9과 동일)
  for modal in [img, lidar, thermal]:
    backbone_feat[modal] = SAM2_encoder(modal)  # with SoftMoE LoRA
    memory_attention(backbone_feat[modal], memory)
    memory.append(backbone_feat[modal])

Phase 2: Auxiliary Prediction (NEW)
  for modal in [img, lidar, thermal]:
    aux_logits[modal] = aux_head(backbone_feat[modal])  # (B, 4, H, W), 공유 head

Phase 3: Energy-based Confidence (NEW - 학습/추론 동일 메커니즘)
  for modal in [img, lidar, thermal]:
    energy[modal] = -torch.logsumexp(aux_logits[modal], dim=1)  # (B, H, W)
    confidence[modal] = -energy[modal].mean(dim=[1, 2])          # (B,), 높을수록 confident
  cross_weights = softmax(stack(confidences) / temperature)       # (B, 3)

Phase 4: UAMM (P9과 동일한 max-norm 방식)
  max_w = cross_weights.max(dim=1, keepdim=True)[0]
  uamm_scores = cross_weights / (max_w + 1e-8)
  modulated_feats = backbone_feats * uamm_scores

Phase 5: Track + AMF (P9과 동일)
  outputs = [track(modulated_feat) for feat in modulated_feats]
  final = sum(cross_weights[i] * outputs[i])
```

---

## 3. 구현 상세

### 3.1 ConfidenceAuxHead (새로 만들 클래스)

파일: `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`

```python
class ConfidenceAuxHead(nn.Module):
    """공유 auxiliary segmentation head. 모든 모달리티가 동일한 head를 사용."""

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        mid_channels = in_channels // 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: backbone feature (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W) — raw logits, softmax 적용 전
        """
        return self.head(feat)
```

**설계 결정:**
- 공유 head 1개 (모달리티별 분리 X) → 파라미터 최소화
- Conv2d 1x1 두 번 = 매우 경량 (in_ch/4 bottleneck)
- BatchNorm 포함 → 모달리티 간 logit scale 정규화 효과
- **softmax 적용하지 않음** — Energy score는 raw logit에서 계산

### 3.2 Energy-based Confidence 계산 (학습 가능 파라미터 없음)

```python
def compute_energy_confidence(aux_logits_list, temperature=1.0):
    """
    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H, W) raw logits
        temperature: energy score temperature (default 1.0)
    Returns:
        weights: (B, num_modalities) — softmax normalized fusion weights
    """
    confidences = []
    for z in aux_logits_list:
        # Energy score: E(x) = -T * log(sum(exp(z_k / T)))
        # 높은 energy (덜 음수) = 더 confident
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)  # (B, H, W)
        conf = -energy.mean(dim=[1, 2])  # (B,), 높을수록 confident
        confidences.append(conf)

    confidences = torch.stack(confidences, dim=1)  # (B, num_modalities)
    weights = F.softmax(confidences / temperature, dim=1)  # (B, num_modalities)
    return weights
```

**핵심 포인트:**
- **학습 가능 파라미터 없음** — weight는 computed signal, 상수로 수렴 불가
- **학습/추론 동일 메커니즘** — P10의 oracle-at-train/guess-at-test 불일치 없음
- **raw logit 기반** — softmax 압축을 우회, backbone 정규화 후에도 차이 유지
- temperature는 하이퍼파라미터 (기본 1.0, config에서 조절 가능)

### 3.3 Expert Collapse Fix (SoftMoE_LoRA_Layer 수정)

파일: `semseg/models/sam2/sam2/sam_lola_utils.py`

`reset_parameters()` 메서드에서 experts_b 초기화만 변경:

```python
def reset_parameters(self):
    nn.init.normal_(self.gate.weight, std=0.01)
    nn.init.zeros_(self.gate.bias)
    if self.cond_dim > 0:
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
    for expert_a in self.experts_a:
        nn.init.kaiming_uniform_(expert_a.weight, a=math.sqrt(5))
    for expert_b in self.experts_b:
        # [P13 변경] zero → small random init로 대칭 깨기
        # 기존: nn.init.zeros_(expert_b.weight)
        nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
        expert_b.weight.data *= 0.01  # scale down → pretrained 모델 근사 유지
```

**설계 결정:**
- `kaiming_uniform_ * 0.01` → 각 expert가 서로 다른 초기값을 가짐 (대칭 깨짐)
- scale 0.01 → LoRA 출력이 pretrained 모델 대비 매우 작음 (초기 동작 거의 동일)
- 이것만으로 gate가 학습 초기부터 의미 있는 gradient를 받음
- **기존 P9 체크포인트와 호환되지 않음** — P13은 처음부터 학습해야 함

---

## 4. LoRA_Sam_P13 클래스 구조

파일: `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`

P9 (`LoRA_Sam_P9`, line 1355)을 복사하여 수정. 변경 최소화.

### P9 대비 변경 목록

| 구분 | P9 | P13 |
| --- | --- | --- |
| 클래스명 | LoRA_Sam_P9 | LoRA_Sam_P13 |
| Head | CrossModalFusionHead | ConfidenceAuxHead (공유 1개) |
| Weight 계산 | head(feats) → softmax | Energy score from aux logits |
| UAMM | max-norm (동일) | max-norm (동일) |
| AMF | cross_weights (동일) | cross_weights (동일) |
| experts_b init | zeros | kaiming * 0.01 |
| 학습 시 반환 | (m_output, m_feat) | (m_output, m_feat, aux_logits_list) |
| 추론 시 반환 | (m_output, m_feat) | (m_output, m_feat) |

### __init__ 변경

```python
class LoRA_Sam_P13(nn.Module):
    def __init__(self, sam_model, rank, num_experts=None, ...):
        super().__init__()
        self.sam = sam_model

        # SoftMoE LoRA (P9과 동일하게 설치, experts_b init만 자동으로 바뀜)
        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()
        # ... (P9과 동일한 LoRA 설치 로직)

        # [P13 NEW] CrossModalFusionHead 대신 ConfidenceAuxHead
        # fusion_dim = backbone 마지막 stage의 차원
        self.aux_head = ConfidenceAuxHead(
            in_channels=fusion_dim,  # Hiera-B+: 896 또는 해당 stage dim
            num_classes=num_classes   # MULTIAQUA: 4
        )

        # [P13 NEW] Energy temperature (config에서 조절 가능)
        self.energy_temperature = 1.0
```

### forward 변경

```python
def forward(self, batched_input, multimask_output):
    # Phase 1: 모달리티별 인코딩 (P9과 동일)
    all_backbone_feats = []
    for modal_idx, modal in enumerate(self.modals):
        backbone_feat = self._encode_modality(batched_input, modal_idx)
        all_backbone_feats.append(backbone_feat)

    # Phase 2: Auxiliary prediction + Energy confidence (NEW)
    aux_logits_list = []
    for feat in all_backbone_feats:
        aux_logits = self.aux_head(feat)  # (B, num_classes, H, W)
        aux_logits_list.append(aux_logits)

    cross_weights = compute_energy_confidence(
        aux_logits_list,
        temperature=self.energy_temperature
    )  # (B, num_modalities)

    # Phase 3: UAMM max-norm (P9과 동일)
    max_w = cross_weights.max(dim=1, keepdim=True)[0]
    uamm_scores = cross_weights / (max_w + 1e-8)

    # 시각화 버퍼 저장 (P9과 동일)
    self._last_uamm_scores = uamm_scores.detach()
    self._last_amf_weights = cross_weights.detach()

    # Phase 4: Feature modulation + Track (P9과 동일)
    # ...

    # Phase 5: AMF (P9과 동일)
    # ...

    if self.training:
        return m_output, m_feat, aux_logits_list  # aux_logits for aux CE loss
    else:
        return m_output, m_feat
```

---

## 5. Loss 함수

### 학습 loss 구성

```python
# 메인 loss (P9과 동일)
seg_loss = OhemCrossEntropy(prediction, gt)
proto_loss = ...  # prototype loss (있으면)

# Aux loss (NEW) — aux head가 각 모달리티에서 정확히 segmentation하도록 학습
aux_loss = 0
for aux_logits in aux_logits_list:
    # backbone feature 해상도에 맞게 GT를 downsample
    gt_downsampled = F.interpolate(gt.float().unsqueeze(1),
                                    size=aux_logits.shape[2:],
                                    mode='nearest').squeeze(1).long()
    aux_loss += F.cross_entropy(aux_logits, gt_downsampled, ignore_index=255)
aux_loss = aux_loss / len(aux_logits_list)

# 총 loss
total_loss = seg_loss + proto_loss + LAMBDA_AUX * aux_loss
```

### LAMBDA_AUX 설정

- 기본값: `0.3` (보조 역할, 메인 loss 지배 방지)
- Config: `LAMBDA_AUX: 0.3`
- P10의 LAMBDA_GATE(0.5)보다 낮게 시작 — aux head는 confidence 계산용이지 메인 목표가 아님

### 중요: fusion weight에는 loss 없음

- **CrossModalFusionHead의 실패 원인이 learned weight였으므로**
- Energy score는 aux_logits에서 computed → aux CE loss만 있으면 됨
- Oracle KL 같은 weight supervision 넣지 않음 (P10 실패 반복 방지)

---

## 6. Config 변경

### 새 config 파일: `configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml`

P9 config에서 변경:

```yaml
MODEL:
  LORA_MODEL    : LoRA_Sam_P13        # P9 → P13
  LORA_R        : 4                    # 동일
  LORA_NUM_EXPERTS : null              # 동일 (auto=3)
  LORA_NUM_CLASSES : 4                 # NEW: aux head용 클래스 수

LOSS:
  LAMBDA_AUX    : 0.3                  # NEW: aux loss 가중치

# energy confidence temperature
ENERGY_TEMPERATURE : 1.0               # NEW
```

나머지 (DATASET, TRAIN, NIGHT_AUG hardaug4, OPTIMIZER, SCHEDULER)는 **P9과 완전히 동일**.

---

## 7. 저장/로드

### save_lora_parameters

P9의 저장 항목에 aux_head 추가:

```python
def save_lora_parameters(self, filename):
    state = {}
    # SoftMoE LoRA params (P9과 동일)
    for name, param in self.sam.named_parameters():
        if 'moe_layer' in name:
            state[name] = param
    # Aux head (NEW)
    for name, param in self.aux_head.named_parameters():
        state[f'aux_head.{name}'] = param
    # prompt_encoder, mask_decoder (P9과 동일)
    # ...
    torch.save(state, filename)
```

### load_lora_parameters

```python
def load_lora_parameters(self, filename):
    state = torch.load(filename)
    # 기존 P9 로직 + aux_head 로드
    for name, param in self.aux_head.named_parameters():
        if f'aux_head.{name}' in state:
            param.data = state[f'aux_head.{name}']
```

---

## 8. 평가 스크립트 호환성

### val_multiaqua.py

- `LoRA_Sam_P13`을 import 목록에 추가
- forward 반환값: 추론 시 (m_output, m_feat)로 P9과 동일 → 기존 평가 로직 호환

### val_multiaqua_P9.py → val_multiaqua_P13.py

- P9 시각화 스크립트를 복사하여 P13용으로 수정
- MoE routing 시각화는 동일하게 작동
- UAMM/AMF 시각화: 이제 이미지별로 다른 값이 나와야 함 (핵심 검증 포인트)

---

## 9. 검증 체크리스트

### 학습 전 확인

- [ ] experts_b 초기화가 비영인지 확인 (print norm)
- [ ] aux_head forward가 정상 작동하는지 확인 (dummy input)
- [ ] Energy confidence가 다른 입력에 대해 다른 weights를 출력하는지 확인

### 학습 중 확인

- [ ] aux_loss가 감소하는지 모니터링
- [ ] UAMM/AMF weights가 이미지별로 다른지 확인 (epoch 1부터)
- [ ] MoE gate weights에서 E1 사용률 > 10%인지 확인 (Block9 기준)
- [ ] val mIoU가 epoch 10 이전에 80% 이상 도달하는지 확인

### 학습 후 확인

- [ ] val_pred/uamm_amf_moe_log.json에서 이미지별 UAMM/AMF 값이 다른지 확인
  - P9: 모든 이미지에서 동일 (img=0.745, lidar=0.961, thermal=1.0)
  - P13: 이미지별로 달라야 함 ← 핵심 성공 기준
- [ ] MoE routing: Block9에서 E1 argmax_fraction > 10%
- [ ] Challenge 제출 (--macvi): M-score > 81.47 (P9 초과) 목표

---

## 10. 요약: P9 → P13 변경 파일 목록

| 파일 | 변경 내용 |
| --- | --- |
| `sam_lora_image_encoder_seg.py` | ConfidenceAuxHead 클래스 추가, LoRA_Sam_P13 클래스 추가 (P9 기반 복사+수정), compute_energy_confidence 함수 추가 |
| `sam_lola_utils.py` | SoftMoE_LoRA_Layer.reset_parameters()에서 experts_b 초기화 변경 |
| `configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml` | 새 config (P9 복사 + LORA_MODEL, LAMBDA_AUX, ENERGY_TEMPERATURE 변경) |
| `train_sam2_lora_paper.py` | P13 import 추가, aux_loss 계산 로직 추가 |
| `val_multiaqua.py` | P13 import 추가 |

### 변경하지 않는 것

- backbone (Hiera-B+), SAM2 memory attention, mask decoder, prompt encoder
- NIGHT_AUG hardaug4 설정
- UAMM max-norm 로직, AMF 로직 (weights 소스만 바뀜)
- 데이터 로더, 평가 metric 계산

---

## 부록: 참고 논문

- **Energy-based OOD Detection** (Liu et al., NeurIPS 2020): Energy score가 softmax confidence 대비 FPR 18% 감소
- **Predictive Dynamic Fusion** (ICML 2024): per-modality confidence 기반 fusion이 generalization error bound를 줄임
- **Useful Confidence Measures: Beyond the Max Score** (NeurIPS 2022 Workshop): entropy/margin이 max probability보다 우수
- **ReliFusion** (2025, Best Paper Canadian AI): confidence-weighted cross-attention fusion

## 부록: Confidence Metric 비교 (Energy 선택 근거)

| Metric | 수식 | softmax 우회 | 추가 파라미터 | 추천도 |
| --- | --- | --- | --- | --- |
| **Energy Score** | `-T * log(Σ exp(z_k/T))` | O (logit 기반) | 없음 | 1순위 |
| Neg. Entropy | `-Σ p_k log p_k` | X (softmax 후) | 없음 | 2순위 |
| Margin | `top1_prob - top2_prob` | X (softmax 후) | 없음 | 3순위 |
| Max Prob | `max(p_k)` | X (softmax 후) | 없음 | 비추천 |

Energy가 1순위인 이유: P9의 CrossModalFusionHead가 상수로 수렴한 핵심 원인이 "softmax/정규화가 모달리티 간 차이를 압축"하는 것. Energy는 softmax 이전의 raw logit magnitude에서 작동하여 이 문제를 근본적으로 우회.
