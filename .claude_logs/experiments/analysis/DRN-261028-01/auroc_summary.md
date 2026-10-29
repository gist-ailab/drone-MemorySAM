# 열화 신호 AUROC 요약

이미지 단위의 6개 점수 중 양방향 AUROC 최대값(점수명). 열화 5종은 전체 case=none과, night/fog는 case=none인 sun과 비교한다.

## hires_s4

| 파티션 | img | depth | event | lidar |
|---|---|---|---|---|
| motionblur | 0.655 (hf_mean) | 0.538 (hf_mean) | 0.551 (norm_std) | 0.514 (cvar_mean) |
| overexposure | 0.788 (cvar_mean) | 0.599 (hf_mean) | 0.532 (cvar_std) | 0.550 (cvar_mean) |
| underexposure | 0.942 (cvar_std) | 0.620 (cvar_mean) | 0.553 (cvar_std) | 0.551 (cvar_mean) |
| lidarjitter | 0.555 (hf_mean) | 0.638 (hf_mean) | 0.564 (hf_std) | 0.624 (cvar_mean) |
| eventlowres | 0.578 (norm_mean) | 0.612 (norm_std) | 0.612 (norm_mean) | 0.591 (norm_std) |
| night | 0.962 (cvar_mean) | 0.559 (norm_mean) | 0.871 (norm_mean) | 0.537 (norm_std) |
| fog | 0.845 (norm_std) | 0.585 (norm_mean) | 0.668 (cvar_std) | 0.550 (norm_mean) |

## block6

| 파티션 | img | depth | event | lidar |
|---|---|---|---|---|
| motionblur | 0.617 (cvar_std) | 0.541 (hf_mean) | 0.551 (hf_std) | 0.514 (norm_mean) |
| overexposure | 0.756 (norm_mean) | 0.587 (hf_mean) | 0.541 (cvar_std) | 0.547 (hf_mean) |
| underexposure | 0.963 (cvar_std) | 0.585 (cvar_mean) | 0.581 (hf_std) | 0.554 (norm_mean) |
| lidarjitter | 0.531 (hf_mean) | 0.622 (cvar_mean) | 0.548 (hf_std) | 0.586 (hf_mean) |
| eventlowres | 0.540 (norm_mean) | 0.577 (cvar_mean) | 0.749 (norm_std) | 0.585 (hf_std) |
| night | 0.996 (norm_std) | 0.550 (hf_mean) | 0.778 (hf_std) | 0.529 (hf_std) |
| fog | 0.992 (cvar_std) | 0.546 (hf_mean) | 0.795 (cvar_std) | 0.528 (hf_std) |
