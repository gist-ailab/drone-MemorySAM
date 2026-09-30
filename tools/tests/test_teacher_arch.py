"""교사 cfg 변환 검사(CPU, 가중치 없이): P56-B·C 학생 cfg 에서 만든 교사 모델의 state_dict 키 집합이
E1 아키텍처(per_modal LoRA, QAF/RangeBias/P55 off)의 키 집합과 정확히 같은지 확인한다."""
import copy, sys, yaml, torch
sys.path.insert(0, '.')
from semseg.models.reliadino.model import build_reliadino

def teacher_cfg(cfg):
    t = copy.deepcopy(cfg)
    t['MODEL'].setdefault('QAF', {})['ENABLE'] = False
    t['MODEL']['LORA_MODE'] = str(cfg['TRAIN'].get('QAF', {}).get('TEACHER_LORA_MODE', 'per_modal'))
    t['MODEL'].pop('LORA_ROUTER', None)
    t['MODEL'].setdefault('FUSION', {}).setdefault('RANGE_BIAS', {})['ENABLE'] = False
    t['MODEL'].setdefault('P55', {})['ENABLE'] = False
    return t

def keys(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['MODEL']['PRETRAINED_BACKBONE'] = False   # 가중치 다운로드 없이 구조만
    torch.manual_seed(0)
    m = build_reliadino(cfg, 25)
    return set(m.state_dict().keys())

e1 = yaml.safe_load(open('configs/hpca100-deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A.yaml'))
e1['MODEL']['QAF']['ENABLE'] = False
ref = keys(e1)
ok = True
for name in ('P56B', 'P56C'):
    cfg = yaml.safe_load(open(f'configs/hpca100-deliver_rgbdel_P46_c3only_seed20260821_screen40_{name}.yaml'))
    stu = keys(cfg)
    tea = keys(teacher_cfg(cfg))
    print(f'{name}: student keys {len(stu)} (E1 대비 +{len(stu-ref)}/-{len(ref-stu)}), teacher keys {len(tea)}, teacher==E1: {tea==ref}')
    ok &= (tea == ref)
print('ALL PASS' if ok else 'FAIL')
