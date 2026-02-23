"""
MULTIAQUA 데이터셋용 Validation 및 Test Inference 스크립트.
- val: validation set 평가 (mIoU, mAcc, Dynamic IoU) + seg/seg_viz 저장. mIoU는 원본 이미지 크기에서 계산.
- test: test set 인퍼런스만 - seg/, seg_viz/ 저장

저장 구조 (val, test 공통):
  save_dir/seg/      : 클래스값 0,1,2,3 (uint8) raw segmentation (원본 크기) - 로컬 평가용
  save_dir/seg_viz/  : Row1 [RGB|Thermal|LiDAR], Row2 [Legend|Seg|Overlay], Row3 [UAMM|AMF|MoE] (LoRA_Sam_P8)
  save_dir/uamm_amf_moe_log.json : 이미지별 UAMM/AMF/MoE LoRA 수치 (LoRA_Sam_P8, --macvi 미사용시)

MaCVi 리더보드 제출 시:
  --macvi 플래그로 실행하면 eval_macvi/에 세그멘테이션 마스크만 1-indexed 저장 (val/test 공통)
  예: python val_multiaqua.py ... --macvi

사용:
  python val_multiaqua.py --cfg configs/lecun_multiaqua_rgbtl_P8.yaml --mode val --model_path outputs/.../epoch15_93.95_checkpoint.pth
  python val_multiaqua.py --cfg ... --mode val --model_path ... --save_dir outputs/.../val_pred
  python val_multiaqua.py --cfg ... --mode test --model_path ... --save_dir outputs/.../test_pred
"""
import torch
import argparse
import yaml
import os
import time
import json
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *


def load_model(cfg, model_path, device):
    """Config 기반 LoRA 모델 로드."""
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']

    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_config_file = "sam2_hiera_b+.yaml"
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        sam2_config_file,
        checkpoint,
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false"
        ]
    )

    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P8')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K')
    lora_layer = model_cfg.get('LORA_LAYER')

    lora_model_class = eval(lora_model_name)
    model_kwargs = {
        'sam_model': sam2,
        'r': lora_r,
        'lora_layer': lora_layer,
    }
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k

    model = lora_model_class(**model_kwargs)

    ckpt = torch.load(str(model_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Model load: {msg}")

    model = model.to(device)
    model.eval()
    return model


def _unpad_resize_to_orig(pred: torch.Tensor, orig_h: int, orig_w: int, model_size: int = 1024) -> torch.Tensor:
    """
    ResizeWidthPadToSquare의 역변환.
    모델 출력 (model_size x model_size)에서 패딩 제거 후 원본 크기로 리사이즈.
    """
    H, W = orig_h, orig_w
    t = model_size
    if W >= H:
        scale = t / W
        nH, nW = round(H * scale), t
        pad_top = (t - nH) // 2
        pad_bottom = t - nH - pad_top
        # pred에서 패딩 제거: [pad_top:pad_top+nH, 0:nW] = (nH, nW)
        pred_content = pred[pad_top : pad_top + nH, :nW]
    else:
        scale = t / H
        nH, nW = t, round(W * scale)
        pad_left = (t - nW) // 2
        pad_right = t - nW - pad_left
        pred_content = pred[:nH, pad_left : pad_left + nW]
    # (nH, nW) -> (orig_h, orig_w)
    if pred_content.shape[0] != H or pred_content.shape[1] != W:
        pred_content = pred_content.unsqueeze(0).unsqueeze(0).float()
        pred_resized = F.interpolate(pred_content, size=(H, W), mode="nearest")
        pred_resized = pred_resized.squeeze(0).squeeze(0).long()
    else:
        pred_resized = pred_content.long()
    return pred_resized


def _draw_legend(classes, palette, target_h, target_w):
    """
    Draw segmentation class color legend. Returns RGB numpy array (target_h x target_w).
    """
    fig, ax = plt.subplots(figsize=(target_w / 80, target_h / 80), dpi=80)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#f8f8f8')

    n = len(classes)
    patch_h = 0.9 / max(n, 1)
    for i, (cls_name, color) in enumerate(zip(classes, palette)):
        if isinstance(color, torch.Tensor):
            color = (color.cpu().numpy() / 255.0).tolist()
        else:
            color = np.asarray(color)
            if color.max() > 1:
                color = (color / 255.0).tolist()
            else:
                color = color.tolist()
        y = 0.95 - (i + 0.5) * patch_h
        rect = plt.Rectangle((0.05, y - patch_h * 0.4), patch_h * 0.8, patch_h * 0.8,
                              facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.05 + patch_h + 0.02, y, cls_name, fontsize=min(18, int(target_h / 35)),
                va='center', ha='left', fontweight='bold')
    ax.set_title('Classes', fontsize=min(22, int(target_h / 30)))
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    from PIL import Image
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _load_modality_image(dataset, modal_key, stem, target_h, target_w):
    """Load single modality image from disk and resize. Returns (H,W,3) uint8."""
    from PIL import Image
    if modal_key == 'img':
        path = dataset.rgb_dir / f"{stem}.png" if hasattr(dataset, 'rgb_dir') else None
    elif modal_key == 'lidar':
        path = dataset.lidar_dir / f"{stem}_lidar.png" if hasattr(dataset, 'lidar_dir') else None
    elif modal_key == 'thermal':
        path = dataset.thermal_dir / f"{stem}_thermal.png" if hasattr(dataset, 'thermal_dir') else None
    else:
        path = None
    if path is None or not path.exists():
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    img = np.array(Image.open(str(path)).convert("RGB"))
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return img


def _draw_bar_chart(values, labels, title, target_h, target_w=None):
    """Draw a horizontal bar chart with value labels. Returns RGB numpy array (target_h x target_w)."""
    fig_w = target_w or max(320, target_h * 2)
    fig, ax = plt.subplots(figsize=(fig_w / 80, target_h / 80), dpi=80)
    n = len(values)
    y_pos = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))
    bars = ax.barh(y_pos, values, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=22)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel('Weight' if 'Fusion' in title else 'Score', fontsize=18)
    # 숫자 값 막대 옆에 표시
    for i, (bar, val) in enumerate(zip(bars, values)):
        txt = f'{val:.3f}' if val < 0.01 or val >= 0.1 else f'{val:.2f}'
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, txt,
                va='center', ha='left', fontsize=18, fontweight='bold')
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))
    img = img[:, :, :3].copy()  # RGB only
    plt.close(fig)
    from PIL import Image
    pil_img = Image.fromarray(img)
    out_w = target_w if target_w else int(img.shape[1] * (target_h / img.shape[0]))
    pil_img = pil_img.resize((out_w, target_h), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def _get_uamm_amf_moe_log(model, batch_idx, modals):
    """
    Extract UAMM, AMF, MoE LoRA values for one sample as a dict for JSON logging.
    Returns dict or None if model has no such attributes.
    """
    core = model.module if hasattr(model, 'module') else model
    if not hasattr(core, '_last_uamm_scores'):
        return None

    uamm = getattr(core, '_last_uamm_scores', None)
    amf = getattr(core, '_last_amf_weights', None)
    moe = getattr(core, '_last_moe_gates', None)

    modal_labels = modals if modals else [f'M{i}' for i in range(uamm.shape[1] if uamm is not None else 0)]
    log = {}

    if uamm is not None and batch_idx < uamm.shape[0]:
        arr = uamm[batch_idx]
        log['uamm'] = {k: round(float(v), 4) for k, v in zip(modal_labels, arr)}

    if amf is not None and batch_idx < amf.shape[0]:
        arr = amf[batch_idx]
        log['amf'] = {k: round(float(v), 4) for k, v in zip(modal_labels, arr)}

    if moe is not None:
        moe_arr = np.asarray(moe)
        if moe_arr.ndim == 1:
            arr = moe_arr
        else:
            arr = moe_arr[batch_idx] if batch_idx < moe_arr.shape[0] else moe_arr[0]
        log['moe'] = {f'E{i}': round(float(v), 4) for i, v in enumerate(arr)}

    return log if log else None


def _get_uamm_amf_moe_viz(model, batch_idx, modals, main_h, main_w):
    """
    Build [UAMM | AMF | MoE] visualization row for one sample.
    - UAMM/AMF: modality labels = config MODALS (img, lidar, thermal) - 각 모달리티별 점수/가중치
    - MoE: expert labels (E0,E1,...) - LoRA expert 선택 비율 (모달리티와 무관)
    Returns single numpy array (viz_h x main_w) or None.
    """
    core = model.module if hasattr(model, 'module') else model
    if not hasattr(core, '_last_uamm_scores'):
        return None

    uamm = getattr(core, '_last_uamm_scores', None)
    amf = getattr(core, '_last_amf_weights', None)
    moe = getattr(core, '_last_moe_gates', None)

    viz_h = int(main_h * 0.55)  # 아래 row 높이: 메인의 55%
    chart_w = (main_w + 2) // 3  # 3개 차트 가로 배분

    modal_labels = modals if modals else [f'M{i}' for i in range(uamm.shape[1] if uamm is not None else 0)]
    strips = []

    if uamm is not None and batch_idx < uamm.shape[0]:
        arr = uamm[batch_idx]
        strips.append(_draw_bar_chart(arr, modal_labels, 'UAMM (Memory Mod)', viz_h, chart_w))

    if amf is not None and batch_idx < amf.shape[0]:
        arr = amf[batch_idx]
        strips.append(_draw_bar_chart(arr, modal_labels, 'AMF (Fusion)', viz_h, chart_w))

    if moe is not None:
        # _last_moe_gates: (num_experts,) - batch dim 없음. (B,num_experts)일 수도 있음.
        moe_arr = np.asarray(moe)
        if moe_arr.ndim == 1:
            arr = np.atleast_1d(moe_arr)  # (num_experts,) -> 전체 사용
        else:
            arr = np.atleast_1d(moe_arr[batch_idx])
        exp_labels = [f'E{i}' for i in range(len(arr))]
        strips.append(_draw_bar_chart(arr, exp_labels, 'MoE LoRA (Experts)', viz_h, chart_w))

    if not strips:
        return None
    bottom = np.concatenate(strips, axis=1)
    # 너비가 main_w와 다를 수 있으므로 리사이즈
    if bottom.shape[1] != main_w:
        from PIL import Image
        bottom = np.array(Image.fromarray(bottom).resize((main_w, viz_h), Image.Resampling.LANCZOS))
    return bottom


def _collate_multiaqua(batch):
    """(sample, label, meta) 배치화. val/test 공통."""
    samples = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    images = [torch.stack([s[i] for s in samples]) for i in range(len(samples[0]))]
    labels = torch.stack(labels)
    return images, labels, metas


@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, macvi_format=False, modals=None):
    """
    Validation 평가. mIoU는 원본 이미지 크기에서 계산.
    macvi_format=False: save_dir/seg/, save_dir/seg_viz/ 생성
      - LoRA_Sam_P8인 경우 seg_viz에 UAMM/AMF/MoE 시각화 함께 저장
    macvi_format=True: save_dir/에 세그멘테이션 마스크만 1-indexed 저장 (시각화 없음)
    """
    from PIL import Image

    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    total_inference_time = 0.0
    num_frames = 0

    if save_dir:
        save_dir = Path(save_dir)
        if macvi_format:
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            seg_dir = save_dir / "seg"
            seg_viz_dir = save_dir / "seg_viz"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_viz_dir.mkdir(parents=True, exist_ok=True)

    modals = modals or getattr(dataloader.dataset, 'modals', ['img', 'lidar', 'thermal'])
    uamm_amf_moe_log = {}  # per-image UAMM/AMF/MoE for JSON

    for images, labels, metas in tqdm(dataloader, desc="Val"):
        images = [x.to(device) for x in images]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, _ = model(images, multimask_output=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0
        num_frames += images[0].shape[0]
        preds = output.softmax(dim=1)
        pred_labels = preds[:, :n_classes].argmax(dim=1)  # (B, H, W)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            orig_label = meta["orig_label"]  # (H, W)
            pred_b = pred_labels[b]  # (model_size, model_size) after ResizeWidthPadToSquare

            # ResizeWidthPadToSquare 역변환: 패딩 제거 후 원본 크기로 리사이즈
            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])

            # mIoU는 원본 크기에서 (pred_resized vs orig_label)
            pred_softmax_orig = F.one_hot(pred_resized.long().clamp(0, n_classes - 1), n_classes).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            metrics.update(pred_softmax_orig, orig_label.unsqueeze(0).to(device))

            if save_dir:
                stem = meta["stem"]
                pred_np = pred_resized.cpu().numpy().astype(np.uint8)  # (orig_h, orig_w)
                if macvi_format:
                    seg_save = (pred_np + 1).clip(1, 4).astype(np.uint8)
                    Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
                else:
                    Image.fromarray(pred_np).save(str(seg_dir / f"{stem}.png"))
                    colored = MULTIAQUA.decode_segmap(pred_np, palette)
                    ds = dataloader.dataset
                    # Layout: Row1 [RGB|Thermal|LiDAR], Row2 [Legend|Seg|Overlay], Row3 [UAMM|AMF|MoE]
                    modality_cols = []
                    for mk in modals:
                        mimg = _load_modality_image(ds, mk, stem, orig_h, orig_w)
                        modality_cols.append(mimg)
                    rgb = modality_cols[0] if modality_cols else np.array(Image.open(str(ds.rgb_dir / f"{stem}.png")).convert("RGB"))
                    if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                        rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                    overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
                    classes = getattr(ds, 'CLASSES', MULTIAQUA.CLASSES)
                    palette = getattr(ds, 'PALETTE', MULTIAQUA.PALETTE)
                    legend_img = _draw_legend(classes, palette, orig_h, orig_w)
                    row1 = np.concatenate(modality_cols, axis=1)
                    row2 = np.concatenate([legend_img, colored, overlay], axis=1)
                    viz_row = np.concatenate([row1, row2], axis=0)
                    viz_bottom = _get_uamm_amf_moe_viz(model, b, modals, viz_row.shape[0], viz_row.shape[1])
                    if viz_bottom is not None:
                        viz_row = np.concatenate([viz_row, viz_bottom], axis=0)
                    Image.fromarray(viz_row).save(str(seg_viz_dir / f"{stem}.png"))
                    # JSON 로깅: 이미지별 UAMM, AMF, MoE LoRA
                    img_log = _get_uamm_amf_moe_log(model, b, modals)
                    if img_log is not None:
                        uamm_amf_moe_log[stem] = img_log

    # save_dir에 uamm_amf_moe_log.json 저장
    if save_dir and not macvi_format and uamm_amf_moe_log:
        log_path = save_dir / "uamm_amf_moe_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "meta": {"modals": modals, "split": "val", "n_images": len(uamm_amf_moe_log)},
                "images": uamm_amf_moe_log,
            }, f, indent=2, ensure_ascii=False)
        print(f"UAMM/AMF/MoE log saved to {log_path}")

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    dynamic_iou = float(ious[1])
    fps = num_frames / total_inference_time if total_inference_time > 0 else 0.0
    return acc, macc, f1, mf1, ious, miou, dynamic_iou, fps


@torch.no_grad()
def run_test_inference(model, dataloader, device, save_dir, macvi_format=False, modals=None):
    """
    Test set 인퍼런스 후 원본 크기로 저장.
    macvi_format=True: eval_macvi/에 세그멘테이션 마스크만 (1-indexed)
    macvi_format=False: seg/, seg_viz/ 생성 (LoRA_Sam_P8일 때 UAMM/AMF/MoE 시각화 포함)
    """
    from PIL import Image

    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE

    save_dir = Path(save_dir)
    if macvi_format:
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        seg_dir = save_dir / "seg"
        seg_viz_dir = save_dir / "seg_viz"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_viz_dir.mkdir(parents=True, exist_ok=True)

    modals = modals or getattr(dataloader.dataset, 'modals', ['img', 'lidar', 'thermal'])
    uamm_amf_moe_log = {}  # per-image UAMM/AMF/MoE for JSON
    idx = 0
    total_inference_time = 0.0
    for images, _, metas in tqdm(dataloader, desc="Test inference"):
        images = [x.to(device) for x in images]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, _ = model(images, multimask_output=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0
        preds = output.softmax(dim=1)
        pred_labels = preds[:, :n_classes].argmax(dim=1)  # (B, H, W)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            stem, orig_h, orig_w = meta["stem"], meta["orig_h"], meta["orig_w"]
            pred_b = pred_labels[b]

            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])
            pred_np = pred_resized.cpu().numpy().astype(np.uint8)

            if macvi_format:
                seg_save = (pred_np + 1).clip(1, 4).astype(np.uint8)
                Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
            else:
                Image.fromarray(pred_np).save(str(seg_dir / f"{stem}.png"))
                colored = MULTIAQUA.decode_segmap(pred_np, palette)
                ds = dataloader.dataset
                modality_cols = []
                for mk in modals:
                    mimg = _load_modality_image(ds, mk, stem, orig_h, orig_w)
                    modality_cols.append(mimg)
                rgb = modality_cols[0] if modality_cols else np.array(Image.open(str(ds.rgb_dir / f"{stem}.png")).convert("RGB"))
                if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                    rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
                classes = getattr(ds, 'CLASSES', MULTIAQUA.CLASSES)
                palette = getattr(ds, 'PALETTE', MULTIAQUA.PALETTE)
                legend_img = _draw_legend(classes, palette, orig_h, orig_w)
                row1 = np.concatenate(modality_cols, axis=1)
                row2 = np.concatenate([legend_img, colored, overlay], axis=1)
                viz_row = np.concatenate([row1, row2], axis=0)
                viz_bottom = _get_uamm_amf_moe_viz(model, b, modals, viz_row.shape[0], viz_row.shape[1])
                if viz_bottom is not None:
                    viz_row = np.concatenate([viz_row, viz_bottom], axis=0)
                Image.fromarray(viz_row).save(str(seg_viz_dir / f"{stem}.png"))
                # JSON 로깅: 이미지별 UAMM, AMF, MoE LoRA
                img_log = _get_uamm_amf_moe_log(model, b, modals)
                if img_log is not None:
                    uamm_amf_moe_log[stem] = img_log
            idx += 1

    # save_dir에 uamm_amf_moe_log.json 저장
    if not macvi_format and uamm_amf_moe_log:
        log_path = save_dir / "uamm_amf_moe_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "meta": {"modals": modals, "split": "test", "n_images": len(uamm_amf_moe_log)},
                "images": uamm_amf_moe_log,
            }, f, indent=2, ensure_ascii=False)
        print(f"UAMM/AMF/MoE log saved to {log_path}")

    fps = idx / total_inference_time if total_inference_time > 0 else 0.0
    if macvi_format:
        print(f"Saved {idx} segmentation masks to {save_dir} (MaCVi 1-indexed)")
    else:
        print(f"Saved {idx} predictions (original size): seg/ and seg_viz/ under {save_dir}")
    print(f"Inference FPS: {fps:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/lecun_multiaqua_rgbtl_P8.yaml')
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--macvi', action='store_true', help='eval_macvi/에 세그멘테이션 마스크만 1-indexed 저장 (val/test 공통)')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(cfg['DEVICE'])
    setup_cudnn()

    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    test_cfg = cfg.get('TEST', {})

    image_size = eval_cfg['IMAGE_SIZE'] if args.mode == 'val' else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    require_annotation = args.mode == 'val'  # val=평가용(annotation필요), test=인퍼런스만(RGB만)
    dataset = MULTIAQUA(
        dataset_cfg['ROOT'],
        split=split,
        transform=transform,
        modals=dataset_cfg['MODALS'],
        require_annotation=require_annotation,
        return_meta=True,
    )
    collate_fn = _collate_multiaqua
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    model = load_model(cfg, model_path, device)

    if args.mode == 'val':
        default_name = "eval_macvi" if args.macvi else "val_pred"
        save_dir = args.save_dir or (model_path.parent / default_name)
        acc, macc, f1, mf1, ious, miou, dynamic_iou, fps = evaluate(
            model, dataloader, device, save_dir=save_dir, macvi_format=args.macvi,
            modals=dataset_cfg.get('MODALS')
        )
        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.2f}" for iou in ious] + [f"{miou:.2f}"],
            'Acc': [f"{a:.2f}" for a in acc] + [f"{macc:.2f}"],
        }
        print("\n" + "=" * 60)
        print(f"MULTIAQUA Validation ({len(dataset)} images)")
        print("=" * 60)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU (original size): {miou:.2f}  mAcc: {macc:.2f}")
        print(f"Dynamic IoU (class 1): {dynamic_iou:.2f}")
        print(f"Inference FPS: {fps:.2f}")
        if save_dir:
            if args.macvi:
                print(f"Saved segmentation masks to {save_dir} (eval_macvi, MaCVi 1-indexed)")
            else:
                print(f"Saved seg/ and seg_viz/ to {save_dir}")

        out_txt = model_path.parent / f"eval_{split}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(out_txt, 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Split: {split}  N={len(dataset)}\n")
            f.write(tabulate(table, headers='keys') + "\n")
            f.write(f"\nDynamic IoU (class 1): {dynamic_iou:.2f}\n")
            f.write(f"Inference FPS: {fps:.2f}\n")
        print(f"Results saved to {out_txt}")

    else:
        default_name = "eval_macvi" if args.macvi else "test_pred"
        save_dir = args.save_dir or (model_path.parent / default_name)
        run_test_inference(
            model, dataloader, device, save_dir, macvi_format=args.macvi,
            modals=dataset_cfg.get('MODALS')
        )


if __name__ == '__main__':
    main()
