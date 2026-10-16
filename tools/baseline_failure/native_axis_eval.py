#!/usr/bin/env python3
"""기준선(DGFusion/CAFuser) 임의 추론 해상도 네이티브 축 재채점 (2026-10-16).

기준선 저장소의 evaluator 는 cfg.DATASETS.DELIVER.EVAL_SIZE(기본 1024)로 예측·GT 를
둘 다 S×S 로 줄여 채점한다. 이 축은 ISSUE-033 에서 우리 모델 기준 +2.56 의 낙관
편향이 확인됐다. 그래서 기준선 해상도 스윕(1024/1280/1536) 비교는 기준선 evaluator 의
자체 숫자가 아니라, 예측을 원본 1042×1042 로 되돌려 우리 legal 프로토콜과 같은 축
(원본 1042² GT · 25클래스 혼동행렬 · common.py 규약)으로 재채점해야 공정하다.

이 스크립트는 기준선 저장소(원격 서버)에 복사해 그 저장소 루트에서 실행되는 독립
파일이다. detectron2·기준선 코드는 지연 import 하며, 이 repo 의 semseg/·val.py 는
import 하지 않는다(거기 없다). 기준선 자체 축(S×S) mIoU 도 공식 evaluator 로 같이
계산해 참고용으로 summary.json 에 남긴다.

DGFusion 공식 가중치 저장소는 train_net.py 가 없고 test_net.py 의 Tester 를 쓴다
(--trainer-module test_net --trainer-class Tester). CAFuser(b) 저장소는 train_net.py
의 Trainer 를 쓴다(기본값). 하드코딩하지 않고 CLI 인자로 흡수한다.

예:
  python native_axis_eval.py \
    --config-file configs/.../dgfusion_deliver.yaml --weights output/.../model.pth \
    --split test --eval-size 1280 --out /ailab_mat2/.../native_axis/dgf_1280 \
    --trainer-module test_net --trainer-class Tester
"""
import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--eval-size", type=int, default=1024,
                    help="추론 해상도 S(예측·GT 를 기준선 evaluator 가 쓰는 S×S). "
                         "네이티브 축은 이와 무관하게 항상 1042² 로 되돌려 채점한다.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer-module", default="train_net",
                    help="Trainer/setup 을 담은 모듈. DGFusion 공식 가중치 저장소는 "
                         "test_net(Tester 가 build_evaluator 를 오버라이드), CAFuser(b) 는 "
                         "train_net(Trainer).")
    ap.add_argument("--trainer-class", default="Trainer",
                    help="build_model/build_test_loader/build_evaluator 를 제공하는 클래스 "
                         "이름. DGFusion 은 Tester, CAFuser(b) 는 Trainer.")
    ap.add_argument("--limit", type=int, default=0,
                    help="디버그용 — 앞 N 장만 처리하고 멈춘다(0=전체).")
    ap.add_argument("--opts", nargs="*", default=[],
                    help="detectron2 config 덮어쓰기(KEY VALUE ...). 기본 opts 뒤에 붙어 "
                         "이긴다.")
    args = ap.parse_args()

    eval_size = args.eval_size

    # 지연 import — 기준선 repo(detectron2 설치)에서만 사용 가능. 스모크(detectron2 부재)
    # 는 여기까지 오지 않고 import·argparse 파싱만 검사한다.
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog

    trainer_mod = importlib.import_module(args.trainer_module)
    TrainerCls = getattr(trainer_mod, args.trainer_class)
    setup = getattr(trainer_mod, "setup")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class _A:
        config_file = args.config_file
        # 기본 opts 먼저, 사용자 --opts 를 뒤에 붙여 사용자가 이기게 한다.
        # MODEL.TEST.DEPTH_ON 은 DGFusion cfg 에만 있고 CAFuser(b) cfg 에는 없는 키라
        # 기본값에 넣지 않는다 — 필요하면 호출 쪽에서 --opts MODEL.TEST.DEPTH_ON False 로 넘긴다.
        opts = ["MODEL.WEIGHTS", args.weights,
                "MODEL.IS_TRAIN", "False",
                "DATASETS.TEST_SEMANTIC", f"('deliver_semantic_{args.split}',)",
                "DATASETS.DELIVER.EVAL_SIZE", str(eval_size),
                "OUTPUT_DIR", str(out_dir / "_d2log")] + list(args.opts)
        eval_only = True
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"

    cfg = setup(_A)
    model = TrainerCls.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.weights, resume=False)
    model.eval()

    ds_name = cfg.DATASETS.TEST_SEMANTIC[0]
    loader = TrainerCls.build_test_loader(cfg, ds_name)

    # DELIVER 매퍼(__call__)가 'sem_seg_file_name'을 dataset_dict.pop()으로 읽어
    # 없애 버리므로(원본 라벨을 읽은 뒤 더는 필요 없다고 보고 제거), 테스트로더가
    # 내보내는 inputs[i]에는 이 키가 없다. 매퍼를 거치기 전의 원본 등록 목록
    # (DatasetCatalog)에서 file_name -> sem_seg_file_name 매핑을 미리 뽑아 둔다.
    raw_dicts = DatasetCatalog.get(ds_name)
    gt_path_by_file = {d["file_name"]: d["sem_seg_file_name"] for d in raw_dicts}

    # 공식(기준선 자체 축 S×S) evaluator — DatasetEvaluators 로 감싸졌을 수도 있으니
    # isinstance 체크 없이 표준 인터페이스(reset/process/evaluate)만 호출한다.
    official_ev = TrainerCls.build_evaluator(
        cfg, ds_name, output_folder=str(out_dir / "official_inference"))
    official_ev.reset()

    # 네이티브 축(원본 1042²) 누적 혼동행렬 — common.confusion_matrix(hist[gt, pred],
    # shape (N, N)) 규약 그대로 더한다.
    native_cm = np.zeros((common.N_CLASSES, common.N_CLASSES), dtype=np.int64)

    n_seen = 0
    with torch.no_grad():
        for bi, inputs in enumerate(loader):
            if args.limit and bi >= args.limit:
                break
            outputs = model(inputs)
            # 공짜로 같은 forward 를 공식 축에도 재사용한다.
            official_ev.process(inputs, outputs)

            for i in range(len(inputs)):
                sem = outputs[i]["sem_seg"]
                pred = sem.argmax(dim=0).cpu().numpy().astype(np.uint8)
                assert pred.shape == (eval_size, eval_size), (
                    f"모델 출력 해상도가 --eval-size 와 다르다: {pred.shape} != "
                    f"({eval_size},{eval_size}) — EVAL_SIZE 패치가 이 체크아웃에 "
                    f"반영됐는지 확인하라")
                pred_1042 = common.resize_nearest(pred, 1042, 1042)

                file_name = inputs[i].get("file_name")
                if file_name is None or file_name not in gt_path_by_file:
                    raise RuntimeError(
                        f"inputs[{i}]['file_name']={file_name!r} 을 raw DatasetCatalog 의 "
                        f"file_name->sem_seg_file_name 매핑에서 찾지 못했다 — 등록 스크립트 "
                        f"(register_deliver_semantic.py)의 키 이름이 다르거나 데이터셋이 "
                        f"다르다(추측 보정 금지).")
                gt_1042 = common.load_gt_deliver(gt_path_by_file[file_name])
                native_cm += common.confusion_matrix(pred_1042, gt_1042)

                n_seen += 1
                if n_seen % 200 == 0:
                    print(f"[native-axis] {n_seen}장 처리", flush=True)

    # 장수 검증 — --limit 미사용 시 기대 장수와 다르면 경고만(에러로 멈추지 않는다).
    if not args.limit:
        exp = common.EXPECTED_COUNTS.get(args.split)
        if exp is not None and n_seen != exp:
            warnings.warn(
                f"[native-axis] split='{args.split}' 처리 장수 {n_seen} != 기대 {exp} — "
                f"로더 필터·경로 규약을 확인하라(계속 진행).")

    _per_class, native_miou = common.global_miou_from_cm(native_cm)
    official_result = official_ev.evaluate()

    summary = {
        "split": args.split,
        "eval_size": eval_size,
        "n_images": n_seen,
        "native_axis_miou": round(native_miou, 4),
        "official_axis_result": official_result,
        "config_file": args.config_file,
        "weights": args.weights,
        "trainer_module": args.trainer_module,
        "trainer_class": args.trainer_class,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[native-axis] split={args.split} eval_size={eval_size} "
          f"NATIVE mIoU={native_miou:.4f} (vs 기준선 자체축 official mIoU 는 "
          f"summary.json 참조)")
    print(f"[native-axis] -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
