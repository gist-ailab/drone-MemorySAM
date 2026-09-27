#!/bin/bash
# DGFusion (b) 런 로그에서 한 줄 요약을 뽑는다: eval 개수, 현재 iter, 마지막 val mIoU,
# 마지막 depth d1, 학습 프로세스 생존 여부. 허브 쪽 추적기가 이 파일을 원격 실행한다.
L=/SSDb/jemo_maeng/dgfusion_train/logs/dgfusion_deliver_degrade_20260920_201226.log
n=$(grep -c "'mIoU':" "$L" 2>/dev/null || echo 0)
it=$(grep -o 'iter: [0-9]*' "$L" 2>/dev/null | tail -1 | grep -o '[0-9]*$')
miou=$(grep -o "'mIoU': [0-9.]*" "$L" 2>/dev/null | tail -1 | grep -o '[0-9.]*$')
# depth 지표 줄은 silog,log10,abs_rel,sq_rel,rms,log_rms,d1,d2,d3 순서라 7번째가 d1 이다.
# 줄 앞에 detectron2 타임스탬프가 붙으므로 행 시작으로 매칭하면 안 된다.
d1=$(grep -A1 'silog,log10,abs_rel' "$L" 2>/dev/null | tail -1 \
     | sed -e 's/\r//' -e 's/.*copypaste: //' | cut -d, -f7)
alive=$(pgrep -cf '[t]rain_net.py.*degrade' 2>/dev/null || echo 0)
echo "${n:-0} ${it:-0} ${miou:-NA} ${d1:-NA} ${alive:-0}"
