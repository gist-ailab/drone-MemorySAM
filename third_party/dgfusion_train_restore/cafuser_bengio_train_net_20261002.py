"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
Training entry restored by merging CAFuser train_net.py (build_train_loader/build_lr_scheduler/build_optimizer + train flow)
into DGFusion test_net.py (which already carries the dgfusion imports and DepthEvaluator wiring).
"""

import copy
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.abspath('./OneFormer'))

from collections import OrderedDict
from typing import Any, Dict, List, Set, Optional, Union

import torch
import warnings
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from oneformer.evaluation import (
    COCOEvaluator,
    DetectionCOCOEvaluator,
    CityscapesInstanceEvaluator,
)

from cafuser.evaluation import (
    MUSESPanopticEvaluator,
    MUSESSemSegEvaluator,
)

from dgfusion.evaluation import DepthEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager

from oneformer import (
    COCOUnifiedNewBaselineDatasetMapper,
    OneFormerUnifiedDatasetMapper,
    InstanceSegEvaluator,
    SemanticSegmentorWithTTA,
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from cafuser import (
    add_cafuser_config,
    add_deliver_config,
)

from dgfusion import (
    MUSESUnifiedDatasetMapper,
    MUSESTestDatasetMapper,
    DELIVERSemanticDatasetMapper,
    add_depth_prediction_config,
)

from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from oneformer.utils.events import WandbWriter, setup_wandb
from time import sleep
from oneformer.data.build import *
from oneformer.data.dataset_mappers.dataset_mapper import DatasetMapper
from PIL import Image

def create_deliver_gt_sem_seg_loading_fn(cfg):
    def deliver_gt_sem_seg_loading_fn(filename: str, copy: bool = False, dtype: Optional[Union[np.dtype, str]] = None) -> np.ndarray:
        with PathManager.open(filename, "rb") as f:
            array = np.array(Image.open(f), copy=copy, dtype=dtype)
        array = array[..., 0]
        array[array != 255] = array[array != 255] - 1
        if sum(sum(array < 0)):
            array[array < 0] = 255

        # This makes the evaluation identical the original DELIVER/CMNeXt codebase (https://github.com/jamycheung/DELIVER)
        if cfg.DATASETS.DELIVER.CMNEXT_EQUIVALENT_EVAL:
            array = array.astype(np.uint8)
            array = np.array(Image.fromarray(array).resize((1024,1024), Image.NEAREST), dtype=dtype)

        return array

    return deliver_gt_sem_seg_loading_fn
    
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DGFusion (training restored from CAFuser train_net.py).
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, inference_only=False):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            if cfg.MODEL.TEST.DETECTION_ON:
                evaluator_list.append(DetectionCOCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # MUSES
        if evaluator_type == "muses_panoptic_seg":
            save_colored_pred=cfg.MODEL.TEST.SAVE_PREDICTIONS.CITYSCAPES_COLORS
            if cfg.MODEL.TEST.PANOPTIC_ON:
                if cfg.MODEL.TEST.SAVE_PREDICTIONS.PANOPTIC or inference_only:
                    panoptic_output_folder = output_folder
                else: 
                    panoptic_output_folder = None
                write_out_confidence=cfg.MODEL.TEST.SAVE_PREDICTIONS.PANOPTIC_CONFIDENCE 
                evaluator_list.append(MUSESPanopticEvaluator(dataset_name, panoptic_output_folder, write_out_confidence, 
                                                                save_colored_pred, inference_only=inference_only))
            if cfg.MODEL.TEST.SEMANTIC_ON:
                evaluator_list.append(MUSESSemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, 
                                                                inference_only=inference_only, save_colored_pred=save_colored_pred))         
            assert not cfg.MODEL.TEST.INSTANCE_ON
            if cfg.MODEL.TEST.DEPTH_ON:
                if cfg.DATASETS.TEST_PANOPTIC == ('muses_panoptic_val',) or not cfg.MODEL.TEST.SAVE_PREDICTIONS.DEPTH:
                    output_folder_depth=None
                else:
                    output_folder_depth = output_folder
                save_depth = cfg.MODEL.TEST.SAVE_PREDICTIONS.DEPTH
                depth_in_log_scale=cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE
                evaluator_list.append(
                    DepthEvaluator(dataset_name, distributed=True, output_dir=output_folder_depth, save_depth_predictions=save_depth,
                                    depth_in_log_scale=depth_in_log_scale))
        # Deliver
        if evaluator_type == "deliver_semantic_seg":
            if cfg.MODEL.TEST.SEMANTIC_ON:
                sem_seg_loading_fn = create_deliver_gt_sem_seg_loading_fn(cfg)
                evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder,
                                                        sem_seg_loading_fn=sem_seg_loading_fn))
            if cfg.MODEL.TEST.PANOPTIC_ON or cfg.MODEL.TEST.INSTANCE_ON:
                raise Exception("The DELIVER segmentation dataset does not support panoptic or instance evaluation")
            if cfg.MODEL.TEST.DEPTH_ON:
                save_depth = cfg.MODEL.TEST.SAVE_PREDICTIONS.DEPTH
                depth_in_log_scale=cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE
                evaluator_list.append(
                    DepthEvaluator(dataset_name, distributed=True, output_dir=output_folder, save_depth_predictions=save_depth,
                                    depth_in_log_scale=depth_in_log_scale, depth_range=(0,255)))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.TEST.DETECTION_ON:
            evaluator_list.append(DetectionCOCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)
    

    @classmethod
    def build_train_loader(cls, cfg):
        # Unified segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "oneformer_unified":
            mapper = OneFormerUnifiedDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco unified segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_unified_lsj":
            mapper = COCOUnifiedNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "muses_unified":
            mapper = MUSESUnifiedDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "deliver_semantic":
            mapper = DELIVERSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)   
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)
    
    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter(),
        ]


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    for p in all_params:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=p.grad)
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST_SEMANTIC
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.INPUT.DATASET_MAPPER_NAME == "muses_unified":
            mapper = MUSESTestDatasetMapper(cfg, False)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "deliver_semantic":
            mapper = DELIVERSemanticDatasetMapper(cfg, False)
        else:
            mapper = DatasetMapper(cfg, False)
        # [baseline_failure A10] BF_EVAL_BATCH — 평가 로더 배치(기본 1 = 공식 그대로).
        # detectron2 build_detection_test_loader 의 batch_size 인자(기본 1)로 넘긴다.
        # 환경변수가 없거나 "1" 이면 아래 if 를 건너뛰고 공식 return 그대로다.
        _bf_raw = os.environ.get("BF_EVAL_BATCH")
        if _bf_raw is not None and _bf_raw.strip() != "1":
            try:
                _bf_eval_batch = int(_bf_raw.strip())
            except ValueError:
                raise ValueError(
                    "BF_EVAL_BATCH 는 1 이상 정수여야 한다: " + repr(_bf_raw))
            if _bf_eval_batch < 1:
                raise ValueError(
                    "BF_EVAL_BATCH 는 1 이상 정수여야 한다: " + repr(_bf_raw))
            print(f"[baseline_failure] BF_EVAL_BATCH={_bf_eval_batch} — "
                  f"평가 로더 배치를 {_bf_eval_batch} 로 올린다")
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper,
                                               batch_size=_bf_eval_batch)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None, eval_only=False, inference_only=False):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST_{TASK}``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        
        if cfg.MODEL.TEST.TASK == "panoptic":
            test_dataset = cfg.DATASETS.TEST_PANOPTIC
        elif cfg.MODEL.TEST.TASK == "instance":
            test_dataset = cfg.DATASETS.TEST_INSTANCE
        elif cfg.MODEL.TEST.TASK == "semantic":
            test_dataset = cfg.DATASETS.TEST_SEMANTIC
        else:
            warnings.warn(f"WARNING: No task provided! Setting task to default value: 'panoptic'")
            test_dataset = cfg.DATASETS.TEST_PANOPTIC

        if evaluators is not None:
            assert len(test_dataset) == len(evaluators), "{} != {}".format(
                len(test_dataset), len(evaluators)
            )
    
        results = OrderedDict

        results = OrderedDict()
        for idx, dataset_name in enumerate(test_dataset):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, inference_only=inference_only)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    add_cafuser_config(cfg)
    add_deliver_config(cfg)
    add_depth_prediction_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only and not args.inference_only:
        setup_wandb(cfg, args)
    # Setup logger for modules
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cafuser")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dgfusion")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="oneformer")
    return cfg


def _maybe_register_degrade_step_hook(cfg, trainer):
    """열화 커리큘럼용 '현재 iteration' 을 데이터로더 워커에 공유하는 훅을 단다.

    데이터로더 워커는 학습 루프의 현재 step 을 직접 알 수 없다. 그래서 학습 쪽에서
    `multiprocessing.Value('i', ...)` 를 만들어 `degradation.set_shared_step` 으로 심고,
    매 iteration 그 값을 갱신한다. 워커는 fork(리눅스 기본 start method)로 이 공유 값을
    물려받으므로, 별도 IPC 없이 현재 severity 상한을 계산할 수 있다.

    🔴 `DEGRADE.ENABLED` 가 꺼져 있으면 아무것도 하지 않는다(기존 재현 런과 완전히 동일).
    DEGRADE 키 자체가 없는(패치 미적용) config 도 꺼짐으로 취급한다.

    호출 순서 주의: 데이터로더 워커는 `trainer.train()` 이 이터레이터를 만들 때 fork 된다.
    따라서 이 함수는 그 전에(= train 호출 전에) 불러 SHARED_STEP 을 미리 심어야 한다.
    """
    try:
        enabled = bool(cfg.DATASETS.DELIVER.DEGRADE.ENABLED)
    except AttributeError:
        enabled = False
    if not enabled:
        return

    import multiprocessing
    from detectron2.engine import HookBase
    # 워커가 mapper 를 통해 import 하는 것과 동일한 모듈 객체여야 SHARED_STEP 이 공유된다.
    from dgfusion.data import degradation as _degradation

    shared_step = multiprocessing.Value("i", int(trainer.start_iter))
    _degradation.set_shared_step(shared_step)

    # 옵션 B(기본 off, ISSUE-041): DEGRADE_STEP_FILE=1 이면 step 을 rank 별 파일로도 공유해
    # spawn 으로 만든 워커도 현재 step 을 읽게 한다. 켜면 커리큘럼 상한이 실제로 0.3→0.6→1.0 으로 진행한다.
    use_file = os.environ.get("DEGRADE_STEP_FILE") == "1"
    if use_file:
        _degradation.enable_step_file(os.path.join(cfg.OUTPUT_DIR, "degrade_step"))
        _degradation.write_step_file(int(trainer.start_iter))

    class _DegradeStepHook(HookBase):
        def before_step(self):
            shared_step.value = int(self.trainer.iter)
            if use_file:
                _degradation.write_step_file(int(self.trainer.iter))

    trainer.register_hooks([_DegradeStepHook()])
    logging.getLogger("dgfusion").info(
        "열화 커리큘럼 활성 — 현재 iteration 을 multiprocessing.Value 로 데이터로더 워커에 공유한다"
    )


def main(args):
    cfg = setup(args)

    if args.inference_only and args.eval_only:
        raise Exception("You can only run inference or evaluation, not both at the same time.")

    elif args.eval_only or args.inference_only:
        model = Trainer.build_model(cfg)
        net_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Params: {} M".format(net_params/1e6))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # [baseline_failure D4/6·A7·A11·A12] BF_ZERO_MODAL 이 있으면 해당 모달 입력 텐서를
        # 채우는 forward pre-hook 을 건다(zero-out ablation). 모델 입력 dict 의 모달 키는
        # CAMERA·LIDAR·EVENT·DEPTH 이고 주 모달(RGB)은 image 키에도 중복 저장된다.
        # CAMERA 를 다룰 때는 image 까지 함께 채운다(두 키가 서로 다른 텐서를 가리킬 수
        # 있으므로 둘 다 처리). 환경변수가 없으면 공식 경로 그대로.
        #
        # 개입 방식(A7·A12) — 환경변수 BF_ZERO_MODE(normalized|raw, 기본 normalized):
        #   normalized: 원본 입력을 **모델 버퍼** model.pixel_mean[3i:3i+3] 으로 채운다.
        #     모델은 모달별 3채널 평균·표준편차를 하나의 긴 버퍼(pixel_mean·pixel_std)로
        #     이어 붙여 갖고 모달 순서 i 로 잘라 (x-mean)/std 정규화를 쓴다(dgfusion.py:
        #     347-348) — 그 슬라이스로 채우면 모델이 보는 값이 0, 우리
        #     modality_zero_ablation(정규화 후 0) 과 같은 개입 규약.
        #   raw: 원본을 0 으로 채운다(옛 동작 — 정규화 후 -PIXEL_MEAN/PIXEL_STD 상수).
        # 모달 순서는 주 모달(cfg.DATASETS.MODALITIES.MAIN_MODALITY) 먼저, 나머지가
        # cfg.DATASETS.MODALITIES.ORDER 순(dgfusion.py:108-132 버퍼 구성과 동일). cfg 의
        # PIXEL_MEAN(A11 탐색)은 모델 슬라이스와의 대조용으로만 쓴다. 버퍼를 못 찾거나
        # 구조가 다르면 에러로 멈춘다(추측 금지).
        #
        # 모달 PIXEL_MEAN 탐색 규칙(A11 — probe_dgfusion.find_modal_pixel_mean 과 동일,
        # 두 곳이 갈라지면 smoke_baseline_failure A11 이 잡는다):
        #   - 후보는 경로 어딘가에 PIXEL_MEAN 토큰쌍이 있고 **마지막 세그먼트**가 모달
        #     이름과 정확히 일치할 때만 인정(부분일치 부정 — DATASETS.PIXEL_MEAN.EVENT_CAMERA
        #     는 CAMERA 의 후보도 EVENT 의 후보도 아니다).
        #   - 예외: PIXEL_MEAN 마커를 떼면 정확히 모달 이름만 남는 무명평탄 키
        #     (LIDAR_PIXEL_MEAN 등), 그리고 CAMERA(주 모달)만 마지막 세그먼트가 그냥
        #     PIXEL_MEAN 인 경로(MODEL.PIXEL_MEAN 등).
        #   - 후보가 여럿이면 우선순위로 하나를 고른다: 경로에 DELIVER(데이터셋 특화)
        #     > 그 외 DATASETS.* > MODEL.* > 기타. 같은 우선순위 안에서 값이 서로 다르면
        #     모호 에러로 멈춘다. 고른 후보와 값이 같은 다른 후보를 로그로 남긴다.
        _bf_zero = os.environ.get("BF_ZERO_MODAL")
        if _bf_zero:
            import torch as _bf_torch
            _bf_modal = _bf_zero.strip().upper()
            if _bf_modal not in ("CAMERA", "LIDAR", "EVENT", "DEPTH"):
                raise ValueError(
                    "BF_ZERO_MODAL 은 CAMERA|LIDAR|EVENT|DEPTH 중 하나여야 한다: "
                    + str(_bf_zero))
            _bf_mode = os.environ.get("BF_ZERO_MODE", "normalized").strip().lower()
            if _bf_mode not in ("normalized", "raw"):
                raise ValueError(
                    "BF_ZERO_MODE 는 normalized|raw 중 하나여야 한다: " + str(_bf_mode))
            _bf_keys = {_bf_modal}
            if _bf_modal == "CAMERA":
                _bf_keys.add("image")   # 주 모달 중복 저장 키(같은 텐서가 아닐 수 있음)

            def _bf_iter_cfg(_node, _prefix=""):
                # cfg(CfgNode·dict·속성 객체) 를 (dotted 경로, 값) 으로 재귀 열거.
                if isinstance(_node, dict):
                    _items = list(_node.items())
                elif hasattr(_node, "keys") and hasattr(_node, "__getitem__"):
                    try:
                        _items = [(str(_k), _node[_k]) for _k in _node.keys()]
                    except Exception:
                        _items = []
                elif hasattr(_node, "__dict__"):
                    _d = {}
                    for _src in (getattr(type(_node), "__dict__", {}) or {},
                                 vars(_node)):
                        for _k, _v in _src.items():
                            if (not _k.startswith("_")) and (not callable(_v)):
                                _d[_k] = _v
                    _items = list(_d.items())
                else:
                    yield _prefix, _node
                    return
                for _k, _v in _items:
                    _p = (_prefix + "." + str(_k)) if _prefix else str(_k)
                    if isinstance(_v, (bool, int, float, str, list, tuple)):
                        yield _p, _v
                    else:
                        yield from _bf_iter_cfg(_v, _p)

            def _bf_seg_has_pixel_mean(_seg):
                # 세그먼트에 PIXEL_MEAN 토큰쌍이 들어가는지(PIXEL_MEAN·LIDAR_PIXEL_MEAN 등,
                # PIXEL_MEANING 같은 우연 일치는 제외).
                _t = _seg.split("_")
                return any(_t[i] == "PIXEL" and _t[i + 1] == "MEAN"
                           for i in range(len(_t) - 1))

            def _bf_strip_pixel_mean_marker(_seg):
                # 세그먼트에서 PIXEL_MEAN 토큰쌍 하나를 떼어 낸 나머지(조인). 쌍이 없으면 None.
                # LIDAR_PIXEL_MEAN->LIDAR, PIXEL_MEAN->"", EVENT_CAMERA->None.
                _t = _seg.split("_")
                for _i in range(len(_t) - 1):
                    if _t[_i] == "PIXEL" and _t[_i + 1] == "MEAN":
                        return "_".join(_t[:_i] + _t[_i + 2:])
                return None

            def _bf_last_seg_is_modal(_last, _modal):
                # A11 — 마지막 세그먼트가 모달 이름과 정확히 일치할 때만 후보.
                _bf_exact = (_last == _modal
                             or (_modal == "CAMERA" and _last == "PIXEL_MEAN")
                             or _bf_strip_pixel_mean_marker(_last) == _modal)
                return _bf_exact

            _bf_prio_names = {3: "데이터셋 특화(DELIVER)", 2: "DATASETS.*",
                              1: "MODEL.*", 0: "기타"}

            def _bf_mean_cand_priority(_up):
                # A11 — 후보 우선순위: 경로에 DELIVER > 그 외 DATASETS.* > MODEL.* > 기타.
                if "DELIVER" in _up:
                    return 3
                _first = _up.split(".")[0]
                if _first == "DATASETS":
                    return 2
                if _first == "MODEL":
                    return 1
                return 0

            def _bf_find_modal_pixel_mean(_cfg, _modal):
                # A11 — probe_dgfusion.find_modal_pixel_mean 과 같은 규칙(부분일치 금지·
                # 우선순위 채택·같은 값 후보 로그). 후보가 없거나 같은 우선순위 안에서
                # 값이 다르면 명확한 에러로 멈춘다(임의값 대입 금지).
                _cands = []
                for _p, _v in _bf_iter_cfg(_cfg):
                    _up = _p.upper()
                    _segs = _up.split(".")
                    if not any(_bf_seg_has_pixel_mean(_s) for _s in _segs):
                        continue
                    if not _bf_last_seg_is_modal(_segs[-1], _modal):
                        continue
                    try:
                        _seq = _v if isinstance(_v, (list, tuple)) else [_v]
                        _m = tuple(float(_c) for _c in _seq)
                    except (TypeError, ValueError):
                        raise RuntimeError(
                            "PIXEL_MEAN 후보(" + _p + ") 값이 숫자가 아니다: " + repr(_v))
                    if not _m:
                        raise RuntimeError(
                            "PIXEL_MEAN 후보(" + _p + ") 가 빈 값이다")
                    _cands.append((_p, _m))
                if not _cands:
                    raise RuntimeError(
                        "cfg 에서 " + _modal + " 의 PIXEL_MEAN 을 찾지 못했다(임의값 대입 "
                        "금지) — config 의 실제 키 이름을 확인한 뒤 다시 실행하라.")
                _top = max(_bf_mean_cand_priority(_p.upper()) for _p, _m in _cands)
                _tier = [(_p, _m) for _p, _m in _cands
                         if _bf_mean_cand_priority(_p.upper()) == _top]
                if len({_m for _p, _m in _tier}) != 1:
                    raise RuntimeError(
                        "모달 " + _modal + " 의 PIXEL_MEAN 후보가 같은 우선순위("
                        + _bf_prio_names[_top] + ") 안에서 값이 달라 모호하다: "
                        + repr(_tier) + " — config 의 모달별 통계 정의를 확인하라(추측 금지).")
                _chosen_p, _mean = _tier[0]
                _same = [_p for _p, _m in _cands if _m == _mean and _p != _chosen_p]
                print("[baseline_failure] " + _modal + " PIXEL_MEAN 후보 "
                      + str(len(_cands)) + "개 — 채택 " + _chosen_p + "("
                      + _bf_prio_names[_top] + "), 값이 같은 다른 후보 "
                      + (repr(_same) if _same else "없음") + " -> " + str(list(_mean)))
                return _mean

            def _bf_modal_order(_cfg):
                # A12 — 모델 pixel_mean 버퍼의 모달 순서: 주 모달(MAIN_MODALITY) 먼저,
                # 나머지는 cfg.DATASETS.MODALITIES.ORDER 순. probe_dgfusion.
                # modal_order_from_cfg 과 같은 규칙(두 곳이 갈라지면 smoke_baseline_
                # failure A7 이 같은 입력으로 잡는다).
                try:
                    _node = _cfg
                    for _k in ("DATASETS", "MODALITIES", "ORDER"):
                        _node = (_node[_k] if isinstance(_node, dict)
                                 else getattr(_node, _k))
                    _order = _node
                    _node = _cfg
                    for _k in ("DATASETS", "MODALITIES", "MAIN_MODALITY"):
                        _node = (_node[_k] if isinstance(_node, dict)
                                 else getattr(_node, _k))
                    _main = _node
                except (AttributeError, KeyError, TypeError):
                    raise RuntimeError(
                        "cfg.DATASETS.MODALITIES.ORDER·MAIN_MODALITY 를 읽을 수 없다 — "
                        "모델 pixel_mean 버퍼의 모달 순서를 알 수 없다(추측 금지). "
                        "BF_ZERO_MODE=raw 로 돌리거나 config 를 확인하라.")
                _mo = [str(_m).strip().upper() for _m in _order]
                _ma = str(_main).strip().upper()
                return [_ma] + [_m for _m in _mo if _m != _ma]

            _bf_mean = None       # A12 — 채움 값(모델 버퍼 슬라이스). raw 면 None.
            _bf_modal_idx = None
            if _bf_mode == "normalized":
                # A12 — 채움 값의 출처는 모델 버퍼, cfg 값(PIXEL_MEAN)은 대조용.
                _bf_modals = _bf_modal_order(cfg)
                if _bf_modal not in _bf_modals:
                    raise RuntimeError(
                        "BF_ZERO_MODAL(" + _bf_modal + ") 이 cfg 모달 순서("
                        + str(_bf_modals) + ") 에 없다 — config 의 DATASETS."
                          "MODALITIES.ORDER·MAIN_MODALITY 를 확인하라(추측 금지).")
                _bf_modal_idx = _bf_modals.index(_bf_modal)
                _bf_pm = getattr(model, "pixel_mean", None)
                if not _bf_torch.is_tensor(_bf_pm):
                    raise RuntimeError(
                        "model.pixel_mean 버퍼가 없다(또는 tensor 가 아니다) — 이 모델의 "
                        "정규화는 dgfusion.py:347-348 구조(모달별 3채널 평균·표준편차를 "
                        "하나의 긴 버퍼로 이어 붙인 pixel_mean·pixel_std)가 아니다. "
                        "BF_ZERO_MODE=raw 로 돌리면 정규화 전 0 채움이 된다(추측 금지).")
                _bf_pm_flat = _bf_pm.detach().float().cpu().reshape(-1)
                if _bf_pm_flat.numel() != 3 * len(_bf_modals):
                    raise RuntimeError(
                        "model.pixel_mean 길이(" + str(_bf_pm_flat.numel())
                        + ") 가 3*모달수(" + str(3 * len(_bf_modals)) + ") 가 아니다 — "
                          "버퍼 구조가 다르다. BF_ZERO_MODE=raw 로 돌리면 정규화 전 0 "
                          "채움이 된다(추측 금지).")
                _bf_slice = _bf_pm_flat[3 * _bf_modal_idx:3 * _bf_modal_idx + 3]
                _bf_mean = tuple(float(_v) for _v in _bf_slice)
                _bf_cfg_mean = _bf_find_modal_pixel_mean(cfg, _bf_modal)
                if not _bf_torch.allclose(
                        _bf_slice,
                        _bf_torch.as_tensor(_bf_cfg_mean, dtype=_bf_torch.float32),
                        atol=1e-4):
                    raise RuntimeError(
                        "모델 버퍼 pixel_mean[" + str(3 * _bf_modal_idx) + ":"
                        + str(3 * _bf_modal_idx + 3) + "]=" + str(list(_bf_mean))
                        + " 와 cfg 평균 " + str(list(_bf_cfg_mean))
                        + " 이 다르다(허용오차 1e-4) — 모달 순서나 통계 정의를 확인하라"
                          "(추측 금지). BF_ZERO_MODE=raw 로 돌리면 정규화 전 0 채움이 된다.")
                print("[baseline_failure] 모달 " + _bf_modal + " 인덱스 "
                      + str(_bf_modal_idx) + ", 모델 평균 " + str(list(_bf_mean))
                      + " = cfg 평균 " + str(list(_bf_cfg_mean))
                      + " — 평균으로 채우면 정규화 후 0")

            def _bf_fill(_v):
                # 원본 텐서를 채널별 모델 버퍼 평균으로 채운다(정규화 후 0 이 되도록).
                _m = _bf_torch.as_tensor(_bf_mean, dtype=_v.dtype,
                                         device=_v.device).reshape(-1)
                if _v.ndim == 3 and _m.numel() == _v.shape[0]:
                    return _bf_torch.ones_like(_v) * _m.view(-1, 1, 1)
                if _m.numel() == 1:
                    return _bf_torch.full_like(_v, float(_m[0]))
                raise RuntimeError(
                    "입력 채널 수(" + str(tuple(_v.shape)) + ") 와 PIXEL_MEAN 길이("
                    + str(_m.numel()) + ") 가 맞지 않아 평균 채움을 정의할 수 없다.")

            # [baseline_failure 부분 열화] BF_ZERO_RATIO 로 모달을 통째로 지우는 대신
            # 원소별로 비율 ratio 만큼만 열화한다. 우리 모델 쪽 규약(tools/
            # missing_modality_eval.py 의 rmm_mask·rmm_degrade)과 같은 의미다 — 정규화
            # 텐서에서 rand<ratio 인 자리를 0(원본 공간에선 그 모달 평균)으로 만들고
            # 픽셀·채널이 서로 독립이다. 기준선은 원본 입력을 다루므로 그 자리를 치환값
            # (normalized 면 모델 평균, raw 면 0)으로 바꾸면 정규화 후 같은 개입이 된다.
            # ratio>=1.0 이면 전부 치환(옛 동작), 0<ratio<1.0 이면 부분 열화한다.
            _bf_ratio = float(os.environ.get("BF_ZERO_RATIO", "1.0"))
            _bf_seed = int(os.environ.get("BF_ZERO_SEED", "0"))
            if _bf_ratio <= 0.0 or _bf_ratio > 1.0:
                raise ValueError(
                    "BF_ZERO_RATIO 는 (0, 1] 범위여야 한다(값을 조용히 고치지 않는다): "
                    + str(_bf_ratio))
            # 우리 도구(missing_modality_eval)처럼 CPU 생성기 하나를 훅 바깥에 두고
            # 호출마다 이어 쓴다 — 훅이 불릴 때마다 새로 만들면 배치마다 같은 마스크가
            # 나와 부분 열화가 편향된다.
            _bf_gen = _bf_torch.Generator().manual_seed(_bf_seed)

            def _bf_zero_hook(_module, _inp):
                bi = _inp[0]
                if isinstance(bi, (list, tuple)):
                    for d in bi:
                        if isinstance(d, dict):
                            for k in _bf_keys:
                                v = d.get(k)
                                if _bf_torch.is_tensor(v):
                                    # 치환값: raw 면 0, normalized 면 모달 평균(정규화 후 0).
                                    filled = (_bf_torch.zeros_like(v)
                                              if _bf_mode == "raw" else _bf_fill(v))
                                    if _bf_ratio >= 1.0:
                                        d[k] = filled   # 전부 치환(옛 동작 그대로)
                                    else:
                                        # rmm_mask 규약 — rand>=ratio 는 유지, 나머지만
                                        # 치환한다(픽셀·채널 독립). CPU 생성기를 이어 쓴다.
                                        keep = (_bf_torch.rand(
                                            v.shape, generator=_bf_gen) >= _bf_ratio
                                            ).to(dtype=v.dtype, device=v.device)
                                        d[k] = v * keep + filled * (1 - keep)
                return _inp
            model.register_forward_pre_hook(_bf_zero_hook)
            _bf_scope = ("전부 치환" if _bf_ratio >= 1.0
                         else f"원소별 비율 {_bf_ratio} 부분 열화")
            if _bf_mode == "normalized":
                print(f"[baseline_failure] 모달 {_bf_modal} 인덱스 {_bf_modal_idx} "
                      f"zero_mode={_bf_mode} ratio={_bf_ratio} seed={_bf_seed} -> keys "
                      f"{sorted(_bf_keys)} 를 모델 버퍼 평균 {list(_bf_mean)} 로 "
                      f"채운다({_bf_scope}, 정규화 후 0)")
            else:
                print(f"[baseline_failure] 모달 {_bf_modal} zero_mode={_bf_mode} "
                      f"ratio={_bf_ratio} seed={_bf_seed} -> keys {sorted(_bf_keys)} 를 "
                      f"0 으로 채운다({_bf_scope}, 정규화 후 -PIXEL_MEAN/PIXEL_STD 상수)")
        # ---------------------------------------------------------------------
        # [baseline_failure NM/EMM-다중모달] 2026-09-23, 판정 세션 승인.
        #
        # 위 BF_ZERO_MODAL 블록(단일 모달, 이미 측정에 쓰인 경로)은 건드리지 않는다.
        # 여기는 별도 메커니즘 둘을 추가한다. 서로 배타적이며, 위 BF_ZERO_MODAL 과도
        # 배타적이다(한 번에 하나만 켠다 — 같이 켜면 에러로 멈춘다. 추측으로 겹쳐 쓰지
        # 않는다).
        #
        #   BF_ZERO_MODALS(콤마구분, 예: "CAMERA,LIDAR") + BF_ZERO_RATIO(기본 1.0)
        #     + BF_ZERO_MODE(normalized|raw, 기본 normalized) + BF_ZERO_SEED
        #     → EMM(ratio=1.0, 결측 조합 전부 채움)·RMM(ratio<1.0, 픽셀·채널 독립
        #       부분 열화)를 **여러 모달 동시에** 적용한다. 개입 규약은 위 단일모달
        #       블록과 완전히 같다(정규화 후 0, rand>=ratio 유지 마스크) — 그냥 모달을
        #       여러 개 동시에 지운다는 차이뿐이다.
        #   BF_NM_TYPE(sp|gaussian) + BF_NM_DENSITY(sp 전용, 기본 0.2)
        #     + BF_NM_STD(gaussian 전용, 기본 0.2) + BF_NM_SEED
        #     → **결측 없이** 존재하는 전 모달에 노이즈를 건다(우리 tools/
        #       missing_modality_eval.py 의 NM 프로토콜과 같은 축 — 재구현이 아니라
        #       tools/baseline_failure/baseline_noise.py 를 통해 그 실제 함수를 그대로
        #       불러 쓴다. 수식 대조는 verify_baseline_noise.py 가 확인했다: S&P·
        #       Gaussian 모두 max|Δ| < 1e-4, 실측 2e-7~2e-5).
        _bf_zero_multi = os.environ.get("BF_ZERO_MODALS")
        _bf_nm_type = os.environ.get("BF_NM_TYPE")
        if _bf_zero_multi and _bf_nm_type:
            raise ValueError(
                "BF_ZERO_MODALS 와 BF_NM_TYPE 를 동시에 켜지 마라(한 번에 하나만"
                "측정한다 — 추측으로 겹쳐 적용하지 않는다).")
        if (_bf_zero_multi or _bf_nm_type) and _bf_zero:
            raise ValueError(
                "BF_ZERO_MODAL(단일)과 BF_ZERO_MODALS/BF_NM_TYPE 를 동시에 켜지 마라.")

        def _bf2_modal_order(_cfg):
            # 위 _bf_modal_order 와 같은 규칙(모델 pixel_mean/std 버퍼의 모달 순서).
            # BF_ZERO_MODAL 이 꺼져 있어도 쓸 수 있게 이 블록 안에 독립 정의한다.
            try:
                _node = _cfg
                for _k in ("DATASETS", "MODALITIES", "ORDER"):
                    _node = (_node[_k] if isinstance(_node, dict) else getattr(_node, _k))
                _order = _node
                _node = _cfg
                for _k in ("DATASETS", "MODALITIES", "MAIN_MODALITY"):
                    _node = (_node[_k] if isinstance(_node, dict) else getattr(_node, _k))
                _main = _node
            except (AttributeError, KeyError, TypeError):
                raise RuntimeError(
                    "cfg.DATASETS.MODALITIES.ORDER·MAIN_MODALITY 를 읽을 수 없다 — "
                    "모델 pixel_mean/std 버퍼의 모달 순서를 알 수 없다(추측 금지).")
            _mo = [str(_m).strip().upper() for _m in _order]
            _ma = str(_main).strip().upper()
            return [_ma] + [_m for _m in _mo if _m != _ma]

        def _bf2_modal_stats(_model, _cfg, _modal):
            # 모델 버퍼(pixel_mean·pixel_std)에서 모달 슬라이스를 뽑는다. 두 버퍼 모두
            # "모달마다 3채널을 이어 붙인 벡터" 구조다(dgfusion.py:347-348 그대로).
            import torch as _t
            _modals = _bf2_modal_order(_cfg)
            if _modal not in _modals:
                raise RuntimeError(
                    "모달 " + _modal + " 이 cfg 모달 순서(" + str(_modals) + ") 에 없다.")
            _idx = _modals.index(_modal)
            _pm = getattr(_model, "pixel_mean", None)
            _ps = getattr(_model, "pixel_std", None)
            if not (_t.is_tensor(_pm) and _t.is_tensor(_ps)):
                raise RuntimeError(
                    "model.pixel_mean/pixel_std 버퍼가 없다 — dgfusion.py:347-348 구조가 "
                    "아니다(추측 금지).")
            _pm_flat = _pm.detach().float().cpu().reshape(-1)
            _ps_flat = _ps.detach().float().cpu().reshape(-1)
            _n = 3 * len(_modals)
            if _pm_flat.numel() != _n or _ps_flat.numel() != _n:
                raise RuntimeError(
                    "pixel_mean/std 길이가 3*모달수(" + str(_n) + ") 가 아니다 — 버퍼 "
                    "구조가 다르다(추측 금지).")
            _mean = _pm_flat[3 * _idx:3 * _idx + 3]
            _std = _ps_flat[3 * _idx:3 * _idx + 3]
            return _idx, _mean, _std

        if _bf_zero_multi:
            import torch as _bf2_torch
            _bf2_modals = [m.strip().upper() for m in _bf_zero_multi.split(",") if m.strip()]
            if not _bf2_modals or any(m not in ("CAMERA", "LIDAR", "EVENT", "DEPTH")
                                       for m in _bf2_modals):
                raise ValueError(
                    "BF_ZERO_MODALS 는 CAMERA|LIDAR|EVENT|DEPTH 의 콤마구분 목록이어야 "
                    "한다(1개 이상): " + str(_bf_zero_multi))
            _bf2_mode = os.environ.get("BF_ZERO_MODE", "normalized").strip().lower()
            if _bf2_mode not in ("normalized", "raw"):
                raise ValueError("BF_ZERO_MODE 는 normalized|raw 중 하나: " + _bf2_mode)
            _bf2_ratio = float(os.environ.get("BF_ZERO_RATIO", "1.0"))
            if _bf2_ratio <= 0.0 or _bf2_ratio > 1.0:
                raise ValueError("BF_ZERO_RATIO 는 (0,1] 범위: " + str(_bf2_ratio))
            _bf2_seed = int(os.environ.get("BF_ZERO_SEED", "0"))
            _bf2_gen = _bf2_torch.Generator().manual_seed(_bf2_seed)

            _bf2_key_to_modal = {}
            for _m in _bf2_modals:
                _bf2_key_to_modal[_m] = _m
                if _m == "CAMERA":
                    _bf2_key_to_modal["image"] = "CAMERA"

            _bf2_fills = {}   # modal -> (C,) 채움값 텐서(raw 모드면 0)
            for _m in _bf2_modals:
                if _bf2_mode == "normalized":
                    _idx, _mean, _std = _bf2_modal_stats(model, cfg, _m)
                    _bf2_fills[_m] = _mean
                    print(f"[baseline_failure/multi] 모달 {_m} 인덱스 {_idx} -> "
                          f"모델 평균 {list(_mean.tolist())} 로 채운다(정규화 후 0)")
                else:
                    _bf2_fills[_m] = None   # raw 모드는 0

            def _bf2_zero_hook(_module, _inp):
                bi = _inp[0]
                if isinstance(bi, (list, tuple)):
                    for d in bi:
                        if isinstance(d, dict):
                            for k, _m in _bf2_key_to_modal.items():
                                v = d.get(k)
                                if _bf2_torch.is_tensor(v):
                                    if _bf2_fills[_m] is None:
                                        filled = _bf2_torch.zeros_like(v)
                                    else:
                                        _mv = _bf2_fills[_m].to(dtype=v.dtype, device=v.device)
                                        filled = (_bf2_torch.ones_like(v) * _mv.view(-1, 1, 1)
                                                  if v.ndim == 3 and _mv.numel() == v.shape[0]
                                                  else _bf2_torch.full_like(v, float(_mv[0])))
                                    if _bf2_ratio >= 1.0:
                                        d[k] = filled
                                    else:
                                        keep = (_bf2_torch.rand(
                                            v.shape, generator=_bf2_gen) >= _bf2_ratio
                                            ).to(dtype=v.dtype, device=v.device)
                                        d[k] = v * keep + filled * (1 - keep)
                return _inp
            model.register_forward_pre_hook(_bf2_zero_hook)
            print(f"[baseline_failure/multi] 모달 {_bf2_modals} ratio={_bf2_ratio} "
                  f"mode={_bf2_mode} seed={_bf2_seed} — EMM/RMM 다중모달 훅 등록")

        if _bf_nm_type:
            import torch as _bf3_torch
            from tools.baseline_failure.baseline_noise import (
                raw_gaussian_noise, raw_sp_noise)

            _bf3_type = _bf_nm_type.strip().lower()
            if _bf3_type not in ("sp", "gaussian"):
                raise ValueError("BF_NM_TYPE 은 sp|gaussian 중 하나: " + _bf_nm_type)
            _bf3_seed = int(os.environ.get("BF_NM_SEED", "0"))
            _bf3_gen = _bf3_torch.Generator().manual_seed(_bf3_seed)
            if _bf3_type == "sp":
                _bf3_density = float(os.environ.get("BF_NM_DENSITY", "0.2"))
                if not (0.0 < _bf3_density < 1.0):
                    raise ValueError("BF_NM_DENSITY 는 (0,1) 범위: " + str(_bf3_density))
                _bf3_param = _bf3_density
            else:
                _bf3_std = float(os.environ.get("BF_NM_STD", "0.2"))
                if _bf3_std <= 0.0:
                    raise ValueError("BF_NM_STD 는 0 초과: " + str(_bf3_std))
                _bf3_param = _bf3_std

            _bf3_modal_order = _bf2_modal_order(cfg)
            _bf3_key_to_modal = {}
            for _m in _bf3_modal_order:
                _bf3_key_to_modal[_m] = _m
                if _m == "CAMERA":
                    _bf3_key_to_modal["image"] = "CAMERA"
            _bf3_stats = {}
            for _m in _bf3_modal_order:
                _idx, _mean, _std = _bf2_modal_stats(model, cfg, _m)
                _bf3_stats[_m] = (_mean, _std)
                print(f"[baseline_failure/nm] 모달 {_m} 인덱스 {_idx} 평균/표준편차 "
                      f"확보(우리 raw_{_bf3_type}_noise 등가식)")

            def _bf3_nm_hook(_module, _inp):
                bi = _inp[0]
                if isinstance(bi, (list, tuple)):
                    for d in bi:
                        if isinstance(d, dict):
                            for k, _m in _bf3_key_to_modal.items():
                                v = d.get(k)
                                if _bf3_torch.is_tensor(v) and v.ndim == 3:
                                    _mean, _std = _bf3_stats[_m]
                                    if _bf3_type == "sp":
                                        noised, _ = raw_sp_noise(
                                            v, _mean, _std, _bf3_param, _bf3_gen)
                                    else:
                                        noised, _ = raw_gaussian_noise(
                                            v, _mean, _std, _bf3_param, _bf3_gen)
                                    d[k] = noised.to(dtype=v.dtype)
                return _inp
            model.register_forward_pre_hook(_bf3_nm_hook)
            print(f"[baseline_failure/nm] type={_bf3_type} param={_bf3_param} "
                  f"seed={_bf3_seed} — 존재 전 모달({_bf3_modal_order})에 NM 훅 등록 "
                  f"(결측 없음, 우리 tools/missing_modality_eval.py 와 같은 함수 재사용)")

        res = Trainer.test(cfg, model, eval_only=args.eval_only, inference_only=args.inference_only)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    # fp16 autocast overflows in OneFormer task_mlp (input-independent task text; its output max grows
    # 52.6k@50k -> 62.9k@80k iters and crosses the fp16 max 65504 near 88k), after which every batch is NaN.
    # bf16 keeps the fp32 exponent range at the same speed/memory. Opt-in so the default stays the official fp16.
    if os.environ.get("DGFUSION_AMP_BF16") == "1" and hasattr(trainer._trainer, "precision"):
        trainer._trainer.precision = torch.bfloat16
        logging.getLogger("detectron2.trainer").info("AMP autocast precision overridden to bfloat16 (DGFUSION_AMP_BF16=1)")
    trainer.resume_or_load(resume=args.resume)
    # 열화 커리큘럼 step 공유(꺼져 있으면 무동작). 데이터로더 워커 fork 전에 심어야 하므로
    # train() 호출 전에 등록하되, **재개(resume) 뒤**에 부른다 — 공유 값의 초기값이
    # trainer.start_iter(재개 지점)여야 재개 직후 미리 읽는 첫 배치들이 그 시점의
    # severity 상한을 쓴다. (jarvis 원본은 resume_or_load 앞이라 start_iter=0 이었다.)
    _maybe_register_degrade_step_hook(cfg, trainer)
    if args.machine_rank == 0:
        net_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print("Total Params: {} M".format(net_params/1e6))
    sleep(3)
    return trainer.train()


if __name__ == "__main__":
    args_parser = default_argument_parser()
    args_parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference on the model.",
    )
    args = args_parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )