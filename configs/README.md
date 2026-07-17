# configs/ — 구조와 규칙

## 폴더 구조

```
configs/
├── deliver/     # DELIVER 데이터셋 학습 config (활성 계열)
├── multiaqua/   # MULTIAQUA 데이터셋 학습 config (활성 계열)
├── det/         # object detection config (det_P*.yaml)
├── eval/        # 평가 전용 config (구 eval_config/ — MODEL_PATH 포함)
├── archive/     # 데드 실험 config (아래 분류 기준 참조)
├── profiles/    # 서버별 경로 참조 문서 (자동 머지 아님)
└── README.md
```

## 분류 기준 (2026-07 재편 시점)

- **활성 계열** (deliver/ · multiaqua/): P8, P9, P22, P27~P31, SAM3RBMA, LoRASam, sam 베이스 계열.
  P22는 adaptive fusion 계열이지만 공동 1위 성능이라 multiaqua/에 유지.
  애매한 버전(P4~P7 등 pre-P8 deliver 계열)은 활성 쪽에 둔다.
- **archive/**: multiaqua adaptive-fusion 데드군 **P10~P21, P23~P26**, 무접두 레거시(`mcubes_*` 등),
  `_REF_*` 참고용 사본.
- **det/**: 기존 위치 그대로 유지 (이동 없음).
- **eval/**: 구 `eval_config/`를 폴더명만 변경. 내부 파일은 그대로.

## ⚠️ 기존 파일명은 바꾸지 않는다

output 디렉토리명이 config 파일명에서 파생되므로(`outputs/<...>/<cfg_name>/...`),
rename은 기존 실험 결과와의 매핑을 깨뜨린다. **재편은 폴더 이동만** 수행했다.

현재 원격 학습 중인 config는 구경로 symlink로 호환 유지:
- `configs/b200-deliver_rgbdel_P31_physaug.yaml` → `deliver/b200-deliver_rgbdel_P31_physaug.yaml`
  (B200에서 학습 중 — 학습 종료 후 symlink 제거 가능)
- `configs/det/det_P31_v3.yaml`은 det/ 위치 불변이므로 symlink 불필요.

## 신규 config 명명 규칙

```
<dataset>_<modal>_<version>_<aug>.yaml
예: deliver_rgbdel_P32_physaug.yaml · multiaqua_rgbtl_P32_hardaug8_physaug.yaml
```

- **서버 접두어 금지** (`b200-`, `levine-` 등 — 기존 파일에만 남아 있는 레거시 관행).
  서버별 차이(데이터 ROOT, PRETRAINED, repo 경로)는 `profiles/<server>.yaml`을 참조해
  config에 수동 반영한다.
- `<modal>`: 사용 모달리티 축약 (rgb/d/e/l/t 조합 — 예: rgbdel, rgbtl).
- `<version>`: 모델 버전 (P8, P31, SAM3RBMA, ...).
- `<aug>`: 증강 프리셋 (hardaug8, physaug, 조합 시 `_`로 연결).

## eval config 대응 규칙

- 학습 config `deliver|multiaqua/<name>.yaml` ↔ 평가 config `eval/<name>.yaml` (동일 파일명).
- eval config는 학습 config + `MODEL_PATH`(체크포인트 경로) 포함형.
- 새 모델 평가 시 학습 config를 eval/로 복사 후 MODEL_PATH만 추가하는 관행 유지.

## profiles/ 사용법

`profiles/<server>.yaml`은 서버별 repo_path·데이터 ROOT·PRETRAINED 관행 경로를 모아 둔
**참조 문서**다. 학습 config에 **자동으로 머지되지 않는다**(자동 머지 = 후속 과제).
신규 config를 특정 서버에서 돌릴 때 이 파일을 보고 ROOT/PRETRAINED를 수동으로 채운다.
서버 메타데이터(ssh alias, GPU 수)의 단일 출처는 `scripts/servers.conf`.
