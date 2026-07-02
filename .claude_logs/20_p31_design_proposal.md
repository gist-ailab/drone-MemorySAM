# P31 재설계 제안 (research_vault 전수 매핑 기반)

> 작성: 2026-07-02. 상태: **proposal (미승인)** — 브레인스토밍 산출물, 승인 후 구현 착수.
> 근거: `16_failure_analysis_P28_P29.md` 정량 진단 + `research_vault/` 96노트 problem→solution 전수 매핑 (seg: S1–S4 / det: D1–D5).
> 구조: **P31-Seg core (novelty 담당)** ↔ **학습 레버 (orthogonal)** ↔ **P31-Det (분리 트랙, 성능 트릭 허용)**.

---

## 0. 설계를 제약하는 확정 사실 (doc 16 정량 진단)

| 사실 | 수치 | 설계 귀결 |
|------|------|------|
| Bridge/Other는 **모든 모달리티가 무능** | modal_competence [0,0,0,0], Bridge IoU 0.0 | **fusion/routing 개선으로 불가** — backbone unfreeze/pretraining만이 지렛대 (ISSUE-008) |
| event/LiDAR reliability **anti-calibrated** | AUROC [img .77, depth .62, **event .30, lidar .22**], drop-Δ [8.4, 23.5, 0.02, 0.01] | 정보가 없는 게 아니라(TrafficLight event .64) **신호가 고장** → 재보정이 진짜 지렛대 |
| DELIVER "야간 갭"의 실체 = **class-transfer (Mode B)** | per-condition spread 2.7–3.6뿐; P29 SDC routing **net −1.1 실증** | 조건 routing 재시도 금지. 허용 레버 = 타깃 증강 + RGB-zero 학습 |
| 진짜 day→night RGB 과의존은 **MULTIAQUA** | val day 93–94 → test night 58–70 | RGB-zeroed dual-loss (2512.17450)가 정조준 |
| 현 P31 프로토타입(HR decoder)은 S1/S3 일부만 커버 | +262K params, GPU 미검증 | 재설계에 흡수 (아래 C 모듈) |

## 0.5 설계 원칙

1. **진단이 지목한 지렛대만 사용** — routing으로 Mode B 풀기, feature-multiply로 dead modality 풀기 같은 기각된 방향 재시도 금지.
2. **novelty는 RBMA(pre-softmax additive reliability bias)를 강화하는 방향으로만 확장** — HyperDUM/PRIMED/SAE/ModalPatch는 재료가 아니라 **ablation 상대/인용 대상**.
3. **det는 head를 빌린다** — absolute mAP는 COCO-pretrained head가 담당, 우리 주장은 "동일 head에서 RGB-only 대비 악조건 delta".

---

## 1. P31-Seg Core — "Calibrated Dual-Reliability RBMA + Multi-scale Class-Token Decoding"

### A. Reliability 재보정 (S2 — 최우선, RBMA를 4모달 전부에서 작동시키기)
- per-modal decoder 용량 증가 + temperature/confidence-penalty 보정 → event/LiDAR B_i AUROC>0.5 달성이 **선행 게이트**.
- 보정 후에만 AMF를 uniform→reliability-proportional로 전환 (현재 uniform [0.27,0.28,0.23,0.23]에 ~45% 질량 낭비).
- 근거: doc 16 §7 처방; UDML(2603.19681)의 raw-entropy "dual suppression" 비판이 왜 지금 고장인지 설명 (vault `43`).

### B. Consistency 2차 bias — RBMA 확장 (S2, novelty 강화)
- `Attention = softmax(QKᵀ/√d + λ_ent·B_ent + λ_cons·B_cons)` — cross-modal 일치도(B_cons)를 **2번째 training-free additive 항**으로 추가.
- 근거: RSGMamba(2604.12319)의 consistency gate를 학습형 baseline으로 인용, 우리는 training-free additive로 차별화 (vault `61`이 이 확장을 명시 제안).
- novelty 서사: "단일 신호 logit-bias → **dual-axis training-free reliability field**"로 RBMA 주장 자체가 강해짐.

### C. Multi-scale 고해상 class-token decoder (S1+S3 — 현 P31 프로토타입 흡수·확장)
- **ViTDet simple-FPN / SAM3-UNet 레시피**(vault `57`)로 RBMA-fused memory feature에서 stride {4,8,16,32} 피라미드 구성 → **고정 per-class 토큰**(Hungarian 없음, Frequency-Matcher 근거: rare 4.8→8.3, vault `51`)이 multi-scale에 cross-attend.
- **training-only auxiliary per-pixel CE head @H/4** (GOOSE-M2F: rare +5–8%, composite +3.4, inference 시 제거, vault `51`) — thin-class gradient 기아 해결.
- 기존 ClassTokenDecoderHR(ConvTranspose ×up)은 pixel-embed 브랜치로 유지.

### D. Complementary reliability-anchored assignment (S1×S2 교차 — 선택 모듈)
- BiXFormer(2506.03675, +22.74 mIoU)의 UMM/CMA 로직을 P30 Router에 이식: **low-reliability 모달의 class query가 강모달이 놓친 mask를 청구**하도록 학습 → event(Ground .37, TrafficLight .64)의 잠재 competence를 실제 기여로 전환.
- ⚠ novelty 충돌 주의: BiXFormer가 "MMSS query decoder" 광역 셀 점유 → 우리 차별 3축 = **reliability-biased SAM2 memory 위 + fused(모달별 아님) + reliability anchor** (vault `59` 방어 논리).

## 2. 학습 레버 (orthogonal — 어느 설계와도 병행, novelty 무충돌)

| 레버 | 해결 | 근거 |
|------|------|------|
| **Backbone 마지막 stage unfreeze / backbone-LoRA** | S1 구조적 dead class (Bridge/Other) — **유일한 지렛대** | doc 16 ISSUE-008 |
| **RGB-zeroed dual-pass loss** `L = L_full + L_rgb-zero` | S4 (MULTIAQUA 야간), S2 보조 | MULTIAQUA(2512.17450, vault `64`) |
| **Class-targeted strong aug** (Wall/TrafficLight/RailTrack-sun/TwoWheeler-cloud) | S4 Mode B — 승인된 유일 레버 | doc 16 §6 |
| **Modality dropout** (det D4와 공유) | S2 + robustness 스토리 | vault `85/87` |
| (장기) ImageNeXt식 pseudo-modal pretraining | S3 천장 65→68 | OmniSegmentor 68.0 (vault `54`) |

## 3. P31-Det — 분리 트랙 (성능 트릭 허용; 19_det_diagnosis_plan Phase 0–1 결과가 전제)

1. **Head 이식 (D1+D3+D5 일괄)**: COCO-pretrained **Deformable-DETR/DINO** head (multi-scale deformable cross-attn — FPN [4,8,16] 전부 샘플링 + DN-denoising)를 RBMA-fused FPN 위에. SAM2 feature 통계≠COCO 기대치 → **projection adapter + warmup** 필수 (vault가 정직하게 "custom multimodal SAM2 backbone에 COCO head 이식한 선행 없음"을 플래그 — 우리가 하면 그 자체가 마이너 기여).
2. **RBMA-in-head (novelty 보존 핵심)**: 빌린 head의 deformable/query cross-attn **pre-softmax logit에 λ·B 주입**. 1차 실험은 inference-time-only (Hungarian 매칭 불안정 회피, vault `15`/`75`: det에서 training-free entropy + additive pre-softmax 셀 **무점유** 확인, MEFormer Eq.9가 실현가능성 선례).
3. **데이터 복원 (D4)**: `REQUIRE_ALL_MODALITIES` 폐지 → per-batch modality dropout, train 5,862→**13,712장**. 평가는 ModalPatch식 10/30/50% drop grid (@50% drop에서 +10~17 mAP 선례) — RBMA delta를 정량화하는 프로토콜로 역이용.
4. **Recipe 수리 (즉시)**: best-ckpt 기준 mAP50로, EMA, ep9-피크-후-하락 LR 원인 수정.
5. **혼동 대응 (D2, E0.3 결과 대기)**: CCF(2603.23276) query-decoupled loss + CBC-SLP class-balanced contrastive. ⚠ vault에 fine-grained 혼동 전용 논문 부재 — E0.3 confusion matrix 확인 전 커밋 금지.

## 4. 우선순위 및 실행 순서 제안

| 순서 | 항목 | 이유 |
|------|------|------|
| ① | Seg-A (재보정) + 레버1 (unfreeze) | 진단이 지목한 두 근본 지렛대. A는 B/D의 전제조건 |
| ② | Seg-C (multi-scale HR class-token) | 현 P31 프로토타입의 자연 확장, S3 최대 기여 예상 |
| ③ | Det-3 (데이터 복원) + Det-4 (recipe) | 학습 전 인프라 — 모든 det 실험의 기반 |
| ④ | Det-1 (head 이식) → Det-2 (RBMA-in-head) | 19번 doc Phase 1 (YOLO 기준점) 결과 확인 후 |
| ⑤ | Seg-B (consistency bias) → Seg-D (complementary) | A 성공(AUROC>0.5) 조건부 |

**Ablation 세트 (논문용)**: RBMA 단일 vs dual-bias / feature-multiply(HyperDUM류) vs logit-bias / uniform vs reliability-proportional AMF / head-only vs head+RBMA-in-head / full-modal vs dropout 학습.

## 5. Novelty 방어 요약

- **채택 금지 (ablation 상대)**: HyperDUM(학습 prototype 가중), PRIMED(learned additive bias — 인접 셀), SAE, ModalPatch(post-softmax multiplicative), MambaFusion(inverse-variance gate).
- **인용 필수 차별화**: BiXFormer(query-MMSS 광역 셀), RSGMamba(learned consistency gate), MEFormer(det additive bias 선례), Decouple-Recouple/WCBR(router 정규화 baseline).
- **생존 claim**: "training-free **dual-axis**(entropy+consistency) reliability를 SAM2 memory attention과 이식된 det head의 **pre-softmax logit**에 일관 주입 — seg·det 공통 프레임워크" — 볼트 확인 기준 4축 조합 무점유.

## 6. 결정 대기 사항

- [ ] P31-Seg core 범위 승인 (A+C 필수 / B+D 조건부 제안)
- [ ] Det head 이식 대상 선택: Deformable-DETR/DINO(연구 유연성) vs RT-DETR/RF-DETR(실시간·성능) — Phase 1 YOLO 결과 참고
- [ ] MULTIAQUA를 4번째 벤치마크로 승격할지 (RGB-zero 학습 스토리와 세트)
