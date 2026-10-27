"""[A2-24GB 2026-10-27] MCubeS 학습 크롭과 동결 평가 변환 CPU 스모크.

실행: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python -I tools/smoke_mcubes_train_crop.py
"""

import ast
import gc
import inspect
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # -I 실행 시 작업 디렉터리가 import 경로에서 빠진다.

from semseg.datasets import mcubes  # noqa: E402


def source_body(source, class_name, method_name=None):
    """AST 노드 본문을 원문 그대로 가져와 HEAD와 비교한다."""
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_name is not None:
        node = next(n for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return ast.get_source_segment(source, node)


def sample_1224x1024():
    """__getitem__의 9개 필드·dtype과 원본 영상 크기를 재현한다."""
    height, width = 1024, 1224
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    label = np.zeros((height, width), dtype=np.uint8)
    # __getitem__은 u/v 좌표맵을 왼쪽 192열로 자른 채 전달한다.
    coord = np.zeros((height, 192), dtype=np.float64)
    return {
        'image': rgb, 'label': label, 'aolp': rgb, 'dolp': rgb,
        'nir': rgb, 'nir_mask': label, 'u_map': coord,
        'v_map': coord, 'mask': label,
    }


def assert_spatial_size(output, size):
    for key, value in output.items():
        assert isinstance(value, torch.Tensor), key
        assert value.shape[-2:] == (size, size), (key, value.shape, size)


def main():
    signature = inspect.signature(mcubes.MCubeS.__init__)
    assert signature.parameters['train_crop'].default == 1024
    assert signature.parameters['train_base'].default is None
    assert signature.parameters['train_crop'].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters['train_base'].kind == inspect.Parameter.KEYWORD_ONLY

    path = ROOT / 'semseg/datasets/mcubes.py'
    current = path.read_text(encoding='utf-8')
    head = subprocess.check_output(
        ['git', 'show', 'HEAD:semseg/datasets/mcubes.py'], cwd=ROOT, text=True)
    for class_name, method_name in [('MCubeS', 'transform_val'), ('FixScaleCrop', None)]:
        assert source_body(current, class_name, method_name) == source_body(head, class_name, method_name), (class_name, method_name)

    sample = sample_1224x1024()
    dataset = mcubes.MCubeS.__new__(mcubes.MCubeS)
    for crop, base in [(768, 768), (1024, 1024), (768, 1024)]:
        dataset.crop_size = crop
        dataset.base_size = base
        # 스케일을 base로 고정해 랜덤 증강의 두 크기와 독립 base 옵션을 검사한다.
        with patch.object(mcubes.random, 'random', return_value=1.0), \
                patch.object(mcubes.random, 'randint', side_effect=[base, 0, 0]):
            train = dataset.transform_tr(sample)
        assert_spatial_size(train, crop)
        del train
        gc.collect()

    # 학습 설정을 768/1024로 둔 상태에서도 평가 출력은 항상 1024다.
    dataset.crop_size = 768
    dataset.base_size = 1024
    val = dataset.transform_val(sample)
    assert_spatial_size(val, 1024)
    print('SMOKE OK')


if __name__ == '__main__':
    main()
