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

    class _DegradeStepHook(HookBase):
        def before_step(self):
            shared_step.value = int(self.trainer.iter)

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
    # 열화 커리큘럼 step 공유(꺼져 있으면 무동작). 데이터로더 워커 fork 전에 심어야 하므로
    # train() 호출 전에 등록한다.
    _maybe_register_degrade_step_hook(cfg, trainer)
    trainer.resume_or_load(resume=args.resume)
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