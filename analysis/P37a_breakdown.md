# Detection breakdown — P37a

checkpoint epoch 4 · train-time metrics {'AP': 0.5853140850308866, 'AP50': 0.8455923033890106, 'AP75': 0.6524137961858272, 'AP_small': 0.18459310807073684, 'AP_medium': 0.4994112169763972, 'AP_large': 0.6908589284921992, 'AR_1': 0.6028664288166269, 'AR_10': 0.6643235937047993, 'AR_100': 0.6799720662769354}
night clips: capture_20260618_114021, capture_20260618_115624

## Overall (mAP / mAP50 / mAP75 — repo reporting convention)

| split | images | mAP | mAP50 | mAP75 | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|---|---|
| all | 3239 | 0.5836 | 0.8449 | 0.6492 | 0.1833 | 0.4977 | 0.6890 |
| night | 1768 | 0.6248 | 0.9098 | 0.6871 | 0.1313 | 0.5361 | 0.6752 |
| normal | 1471 | 0.5584 | 0.7920 | 0.6344 | 0.4133 | 0.4632 | 0.6496 |

**night − normal: mAP50 +0.1178 · mAP +0.0664** (positive = the model holds up better in the dark)

## Per-class

| class | n_gt(all) | AP | AP50 | AP50 night | AP50 normal | night−normal |
|---|---|---|---|---|---|---|
| Allies | 0 | 0.5978 | 0.7690 | 0.9570 | 0.5378 | 0.4192 |
| Enemies | 0 | 0.6669 | 0.8762 | 0.9344 | 0.7849 | 0.1495 |
| Casualties | 0 | 0.7268 | 0.9409 | 0.9898 | 0.8983 | 0.0915 |
| Windows | 0 | 0.6216 | 0.8727 | 0.8890 | 0.8680 | 0.0210 |
| Doors | 0 | 0.6170 | 0.9021 | n/a | 0.9049 | n/a |
| Obstacles | 0 | 0.5175 | 0.6389 | 0.6740 | 0.6138 | 0.0603 |
| Lighting | 0 | 0.5802 | 0.9342 | 0.9310 | 0.9930 | -0.0620 |
| Emergency Exits | 0 | 0.4772 | 0.8359 | 0.9410 | 0.7889 | 0.1521 |
| Fire Extinguishers | 0 | 0.4832 | 0.8937 | 0.8943 | 0.8879 | 0.0063 |
| Landing Markers | 0 | 0.5474 | 0.7857 | 0.9781 | 0.6424 | 0.3358 |

_Classes with n_gt=0 report n/a — absent from this split, not a failure._
