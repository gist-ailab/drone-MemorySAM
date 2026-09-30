"""nm_emm_rmm_block.py 를 train_net.py 의 `res = Trainer.test(` 바로 앞에 끼워 넣은 사본을 만든다.

원본 train_net.py 는 건드리지 않고 `<출력>` 파일만 쓴다(서버마다 train_net.py 계보가 달라 diff 패치가
안 맞는다 — 2026-09-30 yeon 에서 hunk 실패). 삽입 블록은 BF_ZERO_MODAL 블록 뒤(변수 `_bf_zero` 정의 뒤)에
들어가야 하므로 기준선 train_net.py 에 BF_ZERO_MODAL 블록이 이미 있어야 한다.

사용: python apply_bf_block.py <train_net.py> <출력 train_net_bfref.py>
"""
import ast
import sys
from pathlib import Path

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
block = (Path(__file__).parent / "nm_emm_rmm_block.py").read_text(encoding="utf-8").rstrip("\n").split("\n")
lines = src_path.read_text(encoding="utf-8").split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("        res = Trainer.test(cfg, model, eval_only=args.eval_only")]
assert len(idx) == 1, f"앵커(res = Trainer.test) 가 {len(idx)}개다"
assert any("_bf_zero = os.environ.get" in l for l in lines), "BF_ZERO_MODAL 블록이 없다"
out = lines[:idx[0]] + block + lines[idx[0]:]
ast.parse("\n".join(out))
out_path.write_text("\n".join(out), encoding="utf-8")
print(f"삽입 {len(block)}줄 -> {out_path}")
