# Detection module ablation — P37b

baseline mAP50 **0.9485** · night 0.9153 · normal 0.9886

NO-OP criterion: |ΔmAP50| < 0.005 AND top-10 detection agreement > 0.99

| toggle (module OFF) | mAP50 | ΔmAP50 | Δnight | Δnormal | agreement | verdict |
|---|---|---|---|---|---|---|
| p36_router_det_off | 0.9485 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p36_router_off | 0.9485 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p37b_classtoken_det_off | 0.9485 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p39_query_off | 0.9485 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |
| p39_trunkexp_off | 0.9485 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | **NO-OP** |

ACTIVE(+) = turning the module off *hurts* -> the module contributes.
ACTIVE(-) = turning it off *helps* -> the module is a net negative.
NO-OP = the module changes essentially nothing; it is dead weight.
