# Detection breakdown — P37b

checkpoint epoch 5 · train-time metrics {'AP': 0.595010571641577, 'AP50': 0.8401982689399948, 'AP75': 0.6654208401955762, 'AP_small': 0.1573168997730178, 'AP_medium': 0.513744986328359, 'AP_large': 0.7288287415224493, 'AR_1': 0.6182141561836688, 'AR_10': 0.6664875960336383, 'AR_100': 0.6761290509643847}
night clips: capture_20260618_114021, capture_20260618_115624

## Overall (mAP / mAP50 / mAP75 — repo reporting convention)

| split | images | mAP | mAP50 | mAP75 | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|---|---|
| all | 3239 | 0.5932 | 0.8374 | 0.6634 | 0.1558 | 0.5125 | 0.7278 |
| night | 1768 | 0.6269 | 0.8861 | 0.6994 | 0.1283 | 0.5550 | 0.6893 |
| normal | 1471 | 0.5700 | 0.7927 | 0.6408 | 0.3615 | 0.4651 | 0.6982 |

**night − normal: mAP50 +0.0933 · mAP +0.0570** (positive = the model holds up better in the dark)

## Per-class

| class | n_gt(all) | AP | AP50 | AP50 night | AP50 normal | night−normal |
|---|---|---|---|---|---|---|
| Allies | 0 | 0.6259 | 0.7708 | 0.9508 | 0.5491 | 0.4017 |
| Enemies | 0 | 0.7050 | 0.8922 | 0.9275 | 0.8488 | 0.0788 |
| Casualties | 0 | 0.7597 | 0.9498 | 0.9846 | 0.9102 | 0.0744 |
| Windows | 0 | 0.5940 | 0.8734 | 0.9161 | 0.8539 | 0.0622 |
| Doors | 0 | 0.6550 | 0.9305 | n/a | 0.9329 | n/a |
| Obstacles | 0 | 0.4990 | 0.6026 | 0.5959 | 0.6139 | -0.0180 |
| Lighting | 0 | 0.6482 | 0.9337 | 0.9298 | 0.9538 | -0.0239 |
| Emergency Exits | 0 | 0.3920 | 0.7566 | 0.8015 | 0.7451 | 0.0564 |
| Fire Extinguishers | 0 | 0.4909 | 0.8808 | 0.8900 | 0.8816 | 0.0084 |
| Landing Markers | 0 | 0.5620 | 0.7835 | 0.9781 | 0.6381 | 0.3400 |

_Classes with n_gt=0 report n/a — absent from this split, not a failure._
