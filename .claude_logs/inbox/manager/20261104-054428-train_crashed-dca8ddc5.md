---
id: 20261104-054428-train_crashed-dca8ddc5
to: manager
from: pipeline
type: train_crashed
ref: jarvis_jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2L8norouter_20261104_0710
status: done
created: '2026-11-04T05:44:28.720760+00:00'
claimed_by: manager
claimed_at: '2026-11-04T18:14:19.917642'
done_at: '2026-11-04T18:14:19.965561'
note: 15:35 재기동(ep20→ep25 Val 67.05)·17:32 2차 OOM 크래시 확인(동일 유형, 다른 텐넌트 pid). queue.tsv
  a2L8_s902_jarvis 노트에 기록, launched.tsv 정리 완료 — 다음 자동배치 회차가 빈 GPU 생기는 대로 ep25 체크포인트부터
  재개
dedup: 'train_crashed:jarvis_jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2L8norouter_20261104_0710:'
occurrences: 1
last_seen: '2026-11-04T05:44:28.720760+00:00'
---
run_id: jarvis_jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2L8norouter_20261104_0710
이벤트: train_crashed
이유: [rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 42.88 MiB is free. Including non-PyTorch memory, this process has 13.
로그: /home/jemo_maeng/src/drone-MemorySAM-develop/logs/a2L8_s902_jarvis_launch.log
