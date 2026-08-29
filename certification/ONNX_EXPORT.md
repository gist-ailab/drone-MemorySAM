# 드론 탑재용 ONNX (i.MX NPU 대상)

GISTOLO 크로스모달 KD 로 학습한 검출기와 피아식별 분류기를 ONNX 로 내보낸 것이다.
둘 다 **opset 12 · IR 7 · batch 1 고정**이라 NXP eIQ 변환 경로에 그대로 넣을 수 있고,
PyTorch 원본과 수치가 일치함을 확인했다.

| 파일 | 역할 | 크기 | 입력 | 출력 |
|---|---|---|---|---|
| `gistolo_kd_yolov5m_op12.onnx` | 검출 (10클래스) | 80.1 MB | `images` 1×3×640×640 | `output0` 1×25200×15 |
| `iff_mobilenetv3_op12.onnx` | 피아식별 (Allies/Enemies) | 5.8 MB | `input` 1×3×128×128 | `logits` 1×2 |

`.pt` 는 각 ONNX 의 원본 PyTorch 가중치다(재현·재변환용).

## 검증 결과

**검출기** — 노드 337개, 연산자 12종
`Conv · Mul · Sigmoid · Add · Concat · Reshape · MaxPool · Transpose · Split · Pow · Resize · Constant`
PyTorch 대비 최대 절대차 1.16e-03 / 평균 2.13e-06 (allclose atol·rtol 1e-3 통과)

**분류기** — 노드 141개, 연산자 8종
`Conv · Add · Mul · Relu · HardSigmoid · GlobalAveragePool · Flatten · Gemm`
PyTorch 대비 최대 절대차 1.13e-06 (allclose atol 1e-4 통과)

두 모델 모두 NPU 가 흔히 지원하는 연산자만 쓴다. 커스텀 연산자나 NMS 노드는 없다.

## 전처리 / 후처리 (탑재측 구현 필요)

**검출기**
- 입력: BGR→RGB, 640×640 letterbox(패딩값 114), `/255.0`, NCHW. **ImageNet 정규화 없음.**
- 출력 `1×25200×15` = 앵커별 `[cx, cy, w, h, obj, cls0..cls9]`, 좌표는 640 기준 픽셀.
  **NMS 가 그래프에 없으므로 탑재측에서 conf 임계값 + NMS 를 수행해야 한다**
  (평가 시 기본값 conf 0.25 · IoU 0.45).
- 클래스 순서: Allies, Enemies, Casualties, Windows, Doors, Obstacles, Lighting,
  Emergency Exits, Fire Extinguishers, Landing Markers

**분류기**
- 입력: 검출된 사람 박스를 15% 패딩해 크롭 → 128×128 → `/255.0` →
  ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` 정규화.
- 출력 `logits` 1×2 → softmax → `[Allies, Enemies]`.

## 재현

```bash
cd ~/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM-cert/repo
conda activate drone_yolo

# 검출기
python certification/export_det_onnx.py \
  --weights objdet/yolov5m-lowlight/runs_kd_sweep/kd_w003/weights/best.pt \
  --yolo-dir ../third_party/yolov5 \
  --out ../outputs/onnx/gistolo_kd_yolov5m_op12.onnx --img 640 --opset 12

# 피아식별
python certification/export_iff_onnx.py
```

🔴 **opset 주의**: torch 2.9+ 의 기본 dynamo exporter 는 `opset_version=12` 요청을
무시하고 18 로 올린다(Resize 버전 변환 어댑터가 없어 강등도 실패). 두 스크립트 모두
`torch.onnx.export(..., dynamo=False)` 레거시 경로를 명시해 opset 을 고정한다.
yolov5 기본 `export.py` 로 뽑으면 opset 18 이 되니 쓰지 말 것.

## 성능 (참고)

검출기는 GISTOLO 크로스모달 KD 로 학습한 가중치다 — poongsan_v2 test 2,066장에서
mAP50 0.9012 (KD 없는 baseline 0.8963). 피아식별은 정확도 0.8351 (야간 0.8117 /
주간 0.8656). 상세는 상위 폴더 `../kd/README.md` 참조.
