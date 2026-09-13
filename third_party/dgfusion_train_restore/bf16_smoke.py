import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

import test_net

args = default_argument_parser().parse_args([
    "--config-file", "configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml",
    "--eval-only", "MODEL.IS_TRAIN", "False", "MODEL.TEST.DEPTH_ON", "False",
    "OUTPUT_DIR", "output/act_probe",
])
cfg = test_net.setup(args)
model = test_net.Tester.build_model(cfg)
DetectionCheckpointer(model).load("output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth")
model.eval()
tm = {}
model.task_mlp.register_forward_hook(
    lambda m, i, o: tm.__setitem__("max", o.detach().float().abs().max().item()))
loader = test_net.Tester.build_test_loader(cfg, "deliver_semantic_val")
for dtype, tag in [(torch.bfloat16, "BF16"), (torch.float16, "FP16")]:
    fin = []
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for i, b in enumerate(loader):
            out = model(b)
            fin.append(all(torch.isfinite(o["sem_seg"]).all().item() for o in out))
            if i >= 3:
                break
    print(f"{tag} forward ok on 4 imgs, sem_seg finite={all(fin)}, task_mlp max={tm.get('max'):.1f}")
