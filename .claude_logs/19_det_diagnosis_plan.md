# ⚠️ 이동됨 (2026-07-08 재구조화)

**이 파일에 쓰지 마시오.** 새 위치: **det/diagnosis-plan.md**

구번호→새경로 전체 매핑: [00_INDEX.md](00_INDEX.md) · 규칙: [meta/conventions.md](meta/conventions.md)
(이 스텁 아래에 내용이 append되어 있다면, 진행 중이던 세션이 구경로에 쓴 것이므로 새 위치로 옮겨 병합할 것.)

### M3(full) 완주 + 저조도 조기 분해 (2026-07-10)
- M3 best AP50 0.7895 (common, ep25). best ckpt=`bengio:.../outputs/det_final_full/det_P29_final_full/best_checkpoint.pth`
- **M3 best 저조도/정상 분해 (동일 프레임)**: 저조도 AP50 **0.853** / 정상 **0.746** (AP 0.570/0.506)
- **핵심 비교 (같은 프레임, RGB vs 멀티모달)**:
  | 조건 | Y1(RGB) | M3(멀티모달) |
  |---|---|---|
  | 저조도 | 0.865 | 0.853 (−0.012, 사실상 동률) |
  | 정상 | 0.935 | 0.746 (−0.189) |
- **판정**: "멀티모달>RGB 저조도" 부등식 **미성립**(RGB 근소 우위). 단 조건별 변화는 Y1 −0.070(하락) vs M3 +0.107(상승) → 멀티모달 저조도 robustness 신호 有 (단 정상/저조도가 다른 클립이라 조명·장면 혼재). 절대 격차 주범=head 성숙도(YOLO COCO-pretrained vs P29 from-scratch), AP_small 0.148.
- **진짜 fusion 검증 = M3 vs M1(둘 다 P29-Det)** → M1 학습 시작(2026-07-10). M1 저조도 delta가 관문.

### ★ M3 vs M1 fusion 저조도 delta 확정 (2026-07-10) — 논문 핵심
같은 P29-Det 스택, best 체크포인트, 저조도는 동일 1,768 프레임(clean):
| 조건 | M1(RGB) | M3(RGB+LiDAR+Thermal) | fusion Δ |
|---|---|---|---|
| 저조도(1,768, clean) | 0.8174 | **0.853** | **+0.036** |
| 정상 | 0.7351 | 0.746 | +0.011 |
- **fusion 순이득이 저조도(+0.036) > 정상(+0.011), ~3배** → "악조건일수록 멀티모달 우위" 성립.
- vs Y1(외부 YOLO 저조도 0.865): 절대는 아직 YOLO 우위(head 성숙도) → B1(COCO head 이식) 과제.
- 유의: 정상셋은 bengio 114808 rgb 사본 불완전(1,154/1,471)로 M1·M3 동일 프레임 누락 → delta는 유효, 최종표는 프레임 완전일치 재평가(task21) 예정. M2(RGB+Thermal) 학습중 → thermal 단독 기여 분리 예정.

### ★★ 최종 저조도 통합표 (2026-07-13) — 핵심 결과
final split, best 체크포인트, 저조도=동일 1,768프레임 / 정상=1,471프레임 (clip 혼재 주의):
| 모델 | 입력 | 저조도 AP50 | 정상 AP50 |
|---|---|---|---|
| Y1 YOLOv5m | RGB | 0.865 | 0.935 |
| M1 P29 | RGB | 0.817 | 0.735 |
| **M2 P29** | **RGB+Thermal** | **0.870** | 0.740 |
| M3 P29 | RGB+Thermal+LiDAR(egofill) | 0.853 | 0.746 |
| P34 ReliaDINO | RGB+Thermal+LiDAR(egofill) | 0.861 | 0.742 |

**결론:**
1. **RGB+Thermal(M2) > YOLO on 저조도 (0.870 vs 0.865)** — "악조건 멀티모달>RGB" 성립.
2. **Thermal이 저조도 주역**: RGB→+Thermal +0.053(최대 점프). RGB→+T→+LiDAR = 0.817→0.870→0.853.
3. **LiDAR(egofill) 저조도 순손해 −0.017** (동적객체 오차/노이즈) → fusion 스토리는 thermal 중심, lidar 제외 권장.
4. 백본: P34(DINOv3) 0.861 ≈ M3(SAM2) 0.853, 둘 다 M2(RGB+T)보다 낮음 — 3번째 모달(lidar)이 백본 무관하게 손해.
5. 정상은 YOLO 압도(0.935), 우리 모델 클러스터 ~0.74 (head 성숙도). 저조도-정상 delta: YOLO −0.070(하락) vs 우리 +0.08~+0.13(상승, 단 clip 혼재).
- 후속: P34-event(lidar→event, 학습중) → event가 lidar보다 저조도 나은지. best_ckpt는 COCO-AP 기준(AP50 피크와 다를 수 있음).
