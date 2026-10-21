"""[QAF-T 2026-10-21] E1 교사 cfg 구성 회귀 스모크 (CPU, 모델 빌드 없음).

실행: python -I tools/smoke_teacher_cfg.py
"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # -I 에서 작업 디렉터리가 sys.path 에 빠진다.

from train_reliadino import build_teacher_cfg  # noqa: E402


def load_cfg(name):
    with (ROOT / 'configs' / name).open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def legacy_teacher_cfg(cfg, qaf_tc):
    """[A2 2026-10-12] 기존 학생 deepcopy + 교사용 오버라이드 기준."""
    result = copy.deepcopy(cfg)
    model = result['MODEL']
    model.setdefault('QAF', {})['ENABLE'] = False
    model['LORA_MODE'] = str(qaf_tc.get('TEACHER_LORA_MODE', 'per_modal'))
    model.pop('LORA_ROUTER', None)
    model.setdefault('FUSION', {}).setdefault('RANGE_BIAS', {})['ENABLE'] = False
    model.setdefault('P55', {})['ENABLE'] = False
    model.setdefault('HIRES_ADAPTER', {})['ENABLE'] = False
    return result


def main():
    e1_path = ROOT / 'configs/bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_E1.yaml'
    e1 = load_cfg(e1_path.name)
    shared = load_cfg('yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L1shared.yaml')
    shared_before = copy.deepcopy(shared)
    qaf_tc = dict(shared['TRAIN']['QAF'])

    # A: L1 r16/shared 학생이어도 E1 파일의 r8/기본 gated_mlp/M2F 구조를 쓴다.
    qaf_tc['TEACHER_CFG'] = str(e1_path)
    qaf_tc['TEACHER_LORA_R'] = 16  # TEACHER_CFG 사용 시 이 폴백 노브는 무시해야 한다.
    teacher = build_teacher_cfg(shared, qaf_tc)
    model = teacher['MODEL']
    assert model['LORA_MODE'] == 'per_modal'
    assert model['LORA_R'] == e1['MODEL']['LORA_R'] == 8
    assert model['QAF']['ENABLE'] is False
    assert model['HIRES_ADAPTER']['ENABLE'] is False
    assert model['FUSION']['RANGE_BIAS']['ENABLE'] is False
    assert model['P55']['ENABLE'] is False
    assert 'LORA_ROUTER' not in model
    assert model['FUSION'].get('TRUNK', 'gated_mlp') == e1['MODEL']['FUSION'].get('TRUNK', 'gated_mlp')
    assert model['M2F']['ENABLE'] is True
    assert shared == shared_before  # cfg 생성 중 학생 입력이 바뀌지 않는다.

    # B: 파일 지정이 없으면 학생 바탕을 사용하되 교사 rank 기본값 8로 보정한다.
    qaf_tc.pop('TEACHER_CFG')
    qaf_tc.pop('TEACHER_LORA_R')
    model = build_teacher_cfg(shared, qaf_tc)['MODEL']
    assert model['LORA_R'] == 8
    assert model['LORA_MODE'] == 'per_modal'
    assert model['HIRES_ADAPTER']['ENABLE'] is False
    qaf_tc['TEACHER_CFG'] = str(ROOT / 'configs/__missing_teacher_cfg__.yaml')
    assert build_teacher_cfg(shared, qaf_tc)['MODEL']['LORA_R'] == 8

    # C: L2 mean 융합과 L4 M2F 제거가 E1 교사 구조로 전파되지 않는다.
    qaf_tc['TEACHER_CFG'] = str(e1_path)
    mean = load_cfg('yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L2meanfusion.yaml')
    mean_model = build_teacher_cfg(mean, qaf_tc)['MODEL']
    assert mean_model['FUSION'].get('TRUNK', 'gated_mlp') == 'gated_mlp'
    assert mean_model['P39']['TRUNK_EXP'] == e1['MODEL']['P39']['TRUNK_EXP']
    no_m2f = load_cfg('yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L4nom2f.yaml')
    assert build_teacher_cfg(no_m2f, qaf_tc)['MODEL']['M2F']['ENABLE'] is True

    # D: A2 기존 런의 폴백 교사 cfg 는 새 LORA_R=8 이외에 기존 딕셔너리와 동일하다.
    hires = load_cfg('yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hires.yaml')
    hires_qaf_tc = dict(hires['TRAIN']['QAF'])
    hires_qaf_tc.pop('TEACHER_CFG', None)
    actual = build_teacher_cfg(hires, hires_qaf_tc)
    expected = legacy_teacher_cfg(hires, hires_qaf_tc)
    assert actual['MODEL']['LORA_R'] == 8
    actual['MODEL'].pop('LORA_R')
    expected['MODEL'].pop('LORA_R')
    assert actual == expected

    print('SMOKE OK')


if __name__ == '__main__':
    main()
