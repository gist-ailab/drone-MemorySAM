#!/usr/bin/env python3
"""DRN-261105-02 MCubeS 결측 강건성(EMM/RMM, 2503.18445 정의)용 래퍼.
기존 missing_modality_eval 평가기는 무수정으로 사용한다.
val의 데이터셋·모델 로딩·해상도 복원 훅 세 개만 몽키패치한다.
clean 값은 학습기(qualbank_dump) 값과 같아야 한다.
"""
import copy
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# val.py imports the bundled SAM2 package even for a ReliaDINO checkpoint.
_SAM2 = _REPO / "semseg" / "models" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

import val  # noqa: E402
from tools import missing_modality_eval as mme  # noqa: E402
from semseg.datasets.mcubes import MCubeS  # noqa: E402
from semseg.models.reliadino import build_reliadino  # noqa: E402


class _MCubeSMeta(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_classes = dataset.n_classes
        self.ignore_label = dataset.ignore_label
        self.CLASSES = dataset.CLASSES

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample_list, label = self.dataset[i]
        h, w = label.shape[-2:]
        meta = {
            "orig_h": h,
            "orig_w": w,
            "orig_label": label,
            "paths": {"img": str(self.dataset.files[i])},
        }
        return sample_list, label, meta


def _create_dataset(dataset_cfg, split, transform, mode, macvi=False, eval_day=False):
    if str(dataset_cfg["NAME"]).upper() != "MCUBES":
        raise ValueError("mm_eval_mcubes requires DATASET.NAME=MCubeS")
    if split != "val":
        raise ValueError("MCubeS split must be 'val': it reads the 102-image list_folder/test.txt")
    root = os.environ.get("MM_MCUBES_ROOT") or dataset_cfg["ROOT"]
    dataset = MCubeS(root, "val", None, dataset_cfg["MODALS"])
    return _MCubeSMeta(dataset), True


def _load_model(cfg, model_path, device):
    model_cfg = copy.deepcopy(cfg)
    model_cfg["MODEL"]["PRETRAINED_BACKBONE"] = False
    model = build_reliadino(model_cfg, len(MCubeS.CLASSES))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[mm-eval-mcubes] model loaded: missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    if missing:
        raise RuntimeError(f"MCubeS checkpoint has {len(missing)} missing model keys: "
                           f"{sorted(missing)[:12]}")
    if hasattr(model, "_current_epoch"):
        model._current_epoch = 9999
    return model.to(device)


def _unpad_resize_to_orig(pred, orig_h, orig_w, model_size=None):
    if pred.shape[-2:] == (orig_h, orig_w):
        return pred
    raise RuntimeError(
        f"MCubeS prediction shape {tuple(pred.shape[-2:])} differs from "
        f"label shape {(orig_h, orig_w)}; resizing would change the trainer's scoring"
    )


def main():
    val.create_dataset = _create_dataset
    val.load_model = _load_model
    val._unpad_resize_to_orig = _unpad_resize_to_orig
    mme.common.CLASSES = list(MCubeS.CLASSES)
    print("[mm-eval-mcubes] patched val.create_dataset, val.load_model, "
          "val._unpad_resize_to_orig (3 hooks); CSV classes=MCubeS", flush=True)
    mme.main()


if __name__ == "__main__":
    main()
