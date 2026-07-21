# Detection module ablation — P37a

baseline mAP50 **0.9590** · night 0.9236 · normal 0.9885

NO-OP criterion: |ΔmAP50| < 0.005 AND top-10 detection agreement > 0.99

| toggle (module OFF) | mAP50 | ΔmAP50 | Δnight | Δnormal | agreement | verdict |
|---|---|---|---|---|---|---|
| p36_router_det_off | 0.9590 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p36_router_off | 0.9590 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p37a_cefr_off | 0.9393 | -0.0197 | -0.0028 | -0.0244 | 0.0000 | **ACTIVE(+)** |
| p39_query_off | 0.9590 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p39_trunkexp_off | 0.9590 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |

ACTIVE(+) = turning the module off *hurts* -> the module contributes.
ACTIVE(-) = turning it off *helps* -> the module is a net negative.
NO-OP = the module changes essentially nothing; it is dead weight.
