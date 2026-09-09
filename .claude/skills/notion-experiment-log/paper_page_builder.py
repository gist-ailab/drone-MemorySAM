"""논문 페이지(Drone Object Detection for RGB-IR Fusion) 리팩토링 빌더 — 2026-09-08.
절 단위 멱등 갱신: 헤딩이 있으면 replace_section, 없으면 앵커 뒤에 순차 append.
출처 규칙: 노션에는 코드·config·도구 경로와 커밋만 적는다(내부 로그 경로 금지)."""
import os, sys, time
REPO = "/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM"
sys.path.insert(0, os.path.join(REPO, ".claude/skills/notion-experiment-log"))
from notion_api import *  # noqa

PID = "33d05310-a165-408a-b0b8-ec4427d1fe2c"
BM = "3a388e43-5b22-813d-829b-e55ae2ca3a77"       # 📊 벤치마크 (누적)
FIG = os.environ.get("PAPER_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_paper_figs"))
TODAY = "2026-09-08"

def B(s): return rt(s, bold=True)
def T(s): return rt(s)
def P(*parts): return para(*[p if isinstance(p, dict) else rt(p) for p in parts])
def L(*parts): return bul(*[p if isinstance(p, dict) else rt(p) for p in parts])
def CO(text, emoji="💡", color="gray_background"):
    return callout([rt(t) if isinstance(t, str) else t for t in (text if isinstance(text, list) else [text])], emoji, color)
def IMG(name, caption):
    fid = upload(os.path.join(FIG, name))
    if not isinstance(fid, str):
        print("upload failed", name, fid); return P(f"[그림 업로드 실패: {name}]")
    return image_block(fid, caption)

# ───────────────────────────────────────────── 0. 한눈에 보기
def sec_summary():
    return [
        CO([B("연구 대상 "), T("드론·주행 야간/악천후 멀티모달(RGB + LiDAR + Event + Depth/Thermal/Radar) 시맨틱 세그멘테이션과 객체 검출. 벤치 = DELIVER · MUSES · MCubeS(세그) / poongsan indoor(검출) / MULTIAQUA(챌린지, 종료).")], "📌", "blue_background"),
        CO([B("현재 최선(합법 ckpt = val-best 또는 final-iter만, test-best 금지) "),
            T("DELIVER test 54.39±0.76 (5-seed, 최고 단일런 55.29) vs SOTA MM SAM-adapter 57.35 · MUSES 공식 test 79.788 (융합 계보 1위, 카메라단독 1위 GtA 82.39에 −2.60) · MCubeS 58.07±0.49 (published 최고 54.65 대비 +3.42, 1위) · 검출 mAP50 0.9321 (목표 0.85 달성).")], "🏁", "green_background"),
        CO([B("캠페인 결론(2026-06~09) "),
            T("추론 경로 안에서 모달을 적응적으로 가중하는 기제(학습 게이트 · 신뢰도→attention logit bias · 추론 재가중 · 패치별 라우팅 · cross-attention 트렁크 · 인코딩-시간 결합)는 전부 반증됨. 성능을 실제로 움직인 축 = 백본 표현력(SAM2→DINOv3 +11.6) · 학습 해상도(768→1024 test +2.0) · 학습 전용 클래스 prototype 손실(C3, DELIVER +1.4) · 어댑터 정렬 사전학습(+0.74, 재현 대기).")], "🧭", "yellow_background"),
        CO([B("지금 하는 것(2026-09-07~) "),
            T("일일 사이클 실험 카드: 카드 1개 = 변수 1개 = 하루, 40ep 스크린 후 통과 카드만 200ep×3페어 확정. 기준선 B0 legal test 53.78. E1(중간층 4탭 읽기) 트레이너 val ep30 66.88 vs B0 65.13, **E1(중간층 4탭 읽기) legal test 54.85 = B0 +1.07로 카드 첫 스크린 통과**(RailTrack +13.1, Wall·Static 유지) → 200ep×3페어 확정 런 대상. E2(전 선형층 LoRA) 54.50 = +0.72 회색지대(Wall −6.4·Water −5.5 손실, val 순위와 test 순위 역전). E3·E4 완주(legal 대기), E7/E7c 공식 페어 재채점 중, 시드2 페어(B0s2·E1s2·E2s2)·E12 결합·E4b 기동, E7(MUSES PhysAug-off 공정 기준선) 완주. 병행: DGFusion·CAFuser 공식 설정 직접 재학습(재현 수치 확보 후 한계 분석).")], "🔬", "orange_background"),
        P(B("페이지 갱신 규칙 "), T(f"이 페이지는 레포 실험 계획 문서(experiments/plan)과 같은 날짜로 동기화한다. 마지막 동기화 {TODAY}. 절 제목은 고정이고 본문만 교체된다.")),
    ]

# ───────────────────────────────────────────── 1. 문제 정의·목표·평가 프로토콜
def sec_problem():
    return [
        h2("1.1 문제"),
        L("야간·안개·비·눈에서 RGB 단독 인식이 붕괴함 → LiDAR·Event·Depth·Thermal·Radar를 융합해 보완하는 것이 출발점."),
        L("그러나 현 시점 리더보드에서 융합 모델이 카메라 단독(MUSES GtA 82.39)이나 RGB-D 2모달(DELIVER MM SAM-adapter 57.35)을 못 이김 = 융합 접근 자체가 반론 대상."),
        L("우리 측정: 조건×모달 상호작용은 실재함(drop-lidar 손실 day 0.64 vs fog_night 7.2). 그러나 오라클(조건 GT + 전용 가중치)을 줘도 회수분 +0.2 → 남은 격차의 원인은 '배분'이 아니라 '정보·표현'."),
        L("DELIVER의 격차 실체 = 클래스 혼동. RailTrack이 진짜 격차 클래스(우리 4→68로 회복), Wall·Water·Bridge는 DGFusion도 test 0~4로 공통 붕괴."),
        h2("1.2 공식 목표 (2026-07-03 설정)"),
        table([
            ["트랙", "목표", "현재(합법)", "판정"],
            ["Seg — DELIVER (4모달 CLDE)", "논문 publish: SOTA 상회 (MM SAM-adapter val 69.60 / test 57.35)", "5-seed test 54.39±0.76 · best 55.29 · val ≈67.0", "미달 (−2.06~−2.96). DGFusion 자(56.71)로는 best +0.28"],
            ["Seg — MUSES (3모달 CLE)", "SOTA 수준 (GtA 82.39 camera-only / 융합 DGFusion 79.5)", "공식 test 79.788 (val 82.13, 5-seed val 82.03±)", "융합 계보 1위 / 전체 −2.60"],
            ["Seg — MCubeS (4모달)", "3번째 벤치 진입", "3-seed 58.07±0.49", "1위 (+3.42)"],
            ["Det — poongsan indoor", "국가 R&D mAP50 0.85", "0.9321 @ep6 (D1-recovered ViT-L)", "달성 (+0.08)"],
            ["MULTIAQUA (챌린지)", "M-score", "82.10 (P9/P22, 공동 1위)", "종료·고정"],
        ]),
        h2("1.3 평가 프로토콜 (수치를 읽기 전에)"),
        table([
            ["벤치", "합법 수치 정의", "주의"],
            ["DELIVER", "val.py native-GT · BS1 · 하네스 가드(tools/eval_harness_guard.py --check, 8파일 SHA256) · val-best ckpt", "리사이즈-GT 드라이버(ERC)는 +2.56 낙관 → 진단 전용. MM SAM-adapter는 native GT(동일 자), CMNeXt·CAFuser·DGFusion은 1024 리사이즈 GT(낙관 자)"],
            ["MUSES", "Codabench comp 14005 공식 test(단일 제출) / 공식 val = tools/eval_muses_official.py", "제출 게이트: 공식 val ≥ 82.62일 때 1회. val→test 낙차 2.0~3.8 일관"],
            ["MCubeS", "커뮤니티 표준 test split 102장을 로더가 'val'로 읽음 = 트레이너 val이 곧 합법 수치", "val=test라 val-best 선택 = test-best 선택과 동치 → final-epoch 보수치 병기"],
            ["공통", "test-best ckpt 인용 금지 · 헤드라인은 매칭 페어 3시드 이상 · 1pt 미만 차이 주장 금지", "철회 사고 2회(P34 57.60, P46 57.05) 모두 test-best 인용"],
        ]),
        IMG("fig1_sota_gap.png", "3개 벤치에서 SOTA 대비 현재 위치. 양수 = 우리가 앞섬. DELIVER는 5-seed 평균과 최고 단일런을 모두 표기."),
    ]

# ───────────────────────────────────────────── 2. 계보와 가설 원장
def sec_lineage():
    return [
        P("모델 세대별로 '어떤 문제를 풀려고 무엇을 가정했고, 결과가 어땠나'를 한 표로. 약어는 표 안에서 풀어 씀."),
        table([
            ["세대", "겨냥한 문제", "가설(무엇을 바꾸면 오른다고 봤나)", "결과", "판정"],
            ["P8~P27 (SAM2 MemorySAM 기반)", "모달별 품질 차이", "학습형 게이트·Soft-MoE LoRA·spatial quality gating으로 모달 가중", "MULTIAQUA M-score 82.10(공동 1위)이나 DELIVER 미돌파, 게이트가 상수로 수렴", "✗ 학습 게이트 계열 반증 (H1)"],
            ["P28~P33 RBMA", "게이트 붕괴", "training-free 신뢰도(예측 엔트로피)를 SAM2 memory attention의 pre-softmax logit에 additive bias로 주입", "DELIVER val 57.87→64.12(P32-B). attention bias 자체는 순손해(p=4.5e-22)", "✗ (H2). 노벨티 셀은 여전히 비점유이나 성능 근거 없음"],
            ["P34~P36 ReliaDINO", "표현력 부족", "백본을 SAM2 Hiera→frozen DINOv3-L + 모달별 LoRA(Q/V r16)로 교체", "+11.6 (계보 최대 단일 변수). 구 프로토콜 val 68.19/test 56.62. 제안 모듈 ablation 전부 ≈0", "✓ 백본 지배 (H6). 성능 출처 = 백본 + LoRA"],
            ["P37~P39.1", "융합 병목·rank 붕괴", "CEFR 라우팅 / class token / Mask2Former-lite 쿼리 / dual-path / gated-MLP 트렁크 + VICReg(rank 복원)", "CEFR·ClassToken·MaskQuery 무효(no-op). P39.1 rank 수리로 MUSES val 82.62 → 공식 test 79.788(계보 최고)", "✓ rank 복원만 생존. 쿼리 경로 무효 (H9)"],
            ["P40~P45", "MUSES SOTA", "RCA 감쇠 / PanopticDual / BMR 모달 균형", "P40 자기확증 붕괴, P43 79.351, P44 78.429(fog_night 파국)", "✗ 전부 미돌파"],
            ["P46-C3", "DELIVER 클래스 붕괴(RailTrack test 4.02)", "학습 전용 클래스 prototype consistency 손실(추론 그래프 불변)", "RailTrack 4.02→67.69, test +1.35~1.74. MUSES 이식은 −0.77", "✓ 축 특이적 (H8). C1 RCS·C2 MIC는 유해"],
            ["P47~P49", "MUSES 4모달 역전 / RGB 자족", "LiDAR 투영 밀도화 / 모달별 aux CE(UniBal) / MM SAM-adapter식 비대칭 주입", "밀도화 test 78.790(val 과적합), UniBal val 82.06(게이트 82.62 미달), 주입 −0.8~−0.97", "✗ (H11·H13·H14)"],
            ["P50 정렬 사전학습", "어댑터 미정렬(정보 축)", "Places365 pseudo-모달로 frozen 백본 위 어댑터만 cross-modal masked recon", "Phase1 +0.74(페어 1개, 게이트 통과) / 500k 스케일업 −1.85(기각)", "✓ 재현 대기 (H22 강등)"],
            ["P51-CMLC", "추론 선택 불가한 소수 신호", "LoRA 부분공간에서 인코딩-시간 cross-modal 결합", "매칭 페어 Δ −0.82, 악조건일수록 유해(fog −2.03)", "✗ (H19). 융합 기제 4갈래 소거 완성"],
            ["P52 RxDINO", "벤치별 토글 불가 → 단일 config", "C3-adaptive(혼동 EMA→per-class λ) + UniBal-adaptive(손실갭→per-modal λ) + P50 init", "MCubeS seed1 58.18(3-seed 대역 안 = 동률). DELIVER s1/s2 ep38 66.25/66.19. MUSES 보류", "🟡 감사(2026-09-07): 게이트 마진 0.3 < 시드 σ → 판별 불가, P52.1 재설계"],
        ]),
        IMG("fig6_hypothesis_map.png", "가설 원장 H1~H26 판정 맵. 주황 = 반증, 초록 = 확인, 회색 = 미결·예약."),
        h2("2.1 계보가 확립한 명제"),
        L(B("적응 계열은 조건축에서 3단계 전부 닫힘: "), T("융합 가중(H1·H2) → 추론 재가중(H3) → 추출 수준 오라클 상계(H4, 치팅을 줘도 +0.2).")),
        L(B("상호작용은 실재하되 이미 소진됨(H5): "), T("남은 격차의 원인은 배분이 아니라 정보. 이득은 표현력(H6)·해상도(H7)·학습 신호(H8)·데이터에서만 나옴.")),
        L(B("축 특이성(H8·H20): "), T("클래스 prototype 처방은 클래스축 붕괴(DELIVER 심함 +1.4, MCubeS 중간 rubber +9.76·총량 −0.10)에만 통하고 조건축(MUSES) 이식은 −0.77. 사전 등록 예측 2/2 적중 = 진단-구동 배치의 예측력.")),
        L(B("표현력도 L 이상에서 포화(H12): "), T("S+→L +8.82이나 L→H+ +0.52, H+→7B +0.18. '더 큰 백본'은 답이 아님.")),
        L(B("측정 분해능(H18): "), T("DELIVER 5-seed σ 0.76, MUSES 0.92, MCubeS 0.49, 동일 시드 재실행 편차 0.59. 단일런은 대표 불가 → 매칭 페어 3시드 규약.")),
    ]

# ───────────────────────────────────────────── 3. 최근 캠페인 실험 노트
def sec_campaign():
    out = [P("2026-08~09 캠페인의 실험을 '문제 → 가설 → 실험 → 결과 → 판정' 순서로 적는다. 수치는 전부 합법 프로토콜(§1.3).")]
    out += [
        h2("3.1 P46-C3: 학습 전용 클래스 prototype 손실 (DELIVER 클래스 붕괴)"),
        L(B("문제 "), T("DELIVER val→test 전이에서 특정 클래스가 무너짐: RailTrack test 4.02, Wall val 62→test 2, Water·Bridge 저조. 분할 형상은 정성적으로 양호하나 클래스를 헷갈림(2026-07-28 진단).")),
        L(B("가설 "), T("클래스 정체성을 EMA prototype에 당기는 consistency 손실(C3)을 학습에만 추가하면(추론 그래프 불변) 혼동이 줄어든다. 보조 후보: C1 희소클래스 샘플링(RCS), C2 masked consistency(MIC).")),
        L(B("실험 "), T("P39.1 base, DELIVER 4모달, λ∈{0.05, 0.1, 0.15, 0.2, 0.3} 스윕, C1/C2 on/off 격리, 진짜 시드 5개. config configs/jarvis-deliver_rgbdel_P46_ctr_c3only*.yaml, 모델 semseg/models/reliadino/p46.py.")),
        L(B("결과 "), T("C3 단독이 핵심 기제: RailTrack 4.02→67.69, base 대비 test +1.35~1.74. C1은 순유해(RailTrack 64.13 vs C1+C3 59.10), C2는 −1.67. λ 0.05~0.2 평탄. 5-seed 정본(legal-val 재선택 후) test 54.39±0.76, best seed816 55.29.")),
        L(B("판정 "), T("H8 ✓ (클래스축 특이). C1·C2 폐기(H15 ✗). MUSES 이식(λ0.2) 공식 test 79.023 = seed2 −0.765, 손해가 clear/day에 집중 → 축이 다른 벤치에 처방 이식 금지.")),
        CO([B("철회 기록 — "), T("2026-08-03 'DELIVER test 57.05 = SOTA 돌파'는 test-best ckpt(ep108) 수치였음. 합법 재계산 최고 55.62~55.69로 DGFusion 56.71 미달. 2026-08-04 정정. 무효 수치는 참고용으로만 보관.")], "🚩", "red_background"),
        h2("3.2 측정기 대정비(ISSUE-033)와 시드 분산"),
        L(B("문제 "), T("헤드라인 val 69.44/test 56.99가 리사이즈-GT 낙관 드라이버 산으로 밝혀짐(+2.56). 시드 편차는 미측정이라 +0.3 수준 판정이 노이즈였음.")),
        L(B("조치 "), T("정본 = val.py native-GT BS1 + 하네스 가드(tools/eval_harness_guard.py, 8파일 SHA256 동결). 과거 합법 수치 전수 재확인. N6: 5런 legal-val 재선택 → mean 53.82→54.39±0.76, 선택 아티팩트 2/5런 실재.")),
        L(B("판정 "), T("H18 ✗(단일런 대표 불가). 규약: 헤드라인·게이트는 매칭 페어 3시드 이상, '기준 −0.3 이내 통과' 형식 금지, 온라인 가중 실험은 λ 궤적 플롯 제출 의무.")),
        IMG("fig5_power.png", "검정력 계산: 시드 2~3개로 판별 가능한 차이는 페어 설계 ±1.0, 독립 설계 ±2 이상. 구 P52 게이트 마진 0.3은 팔당 34~138 시드가 필요했다."),
        h2("3.3 MCubeS 진입과 dose-response (진단-구동 처방의 예측력)"),
        L(B("문제 "), T("C3의 효과가 '사후 설명'인지 '예측'인지 구분 필요. 리뷰어 반론 = 벤치 간 상관은 인과가 아님.")),
        L(B("가설(사전 등록) "), T("C3 효과는 class-transfer 붕괴 강도에 단조: DELIVER(심함) +1.4 / MUSES(없음) −0.77 → MCubeS(중간: rubber 등)에서 표적 클래스 회복 + 총량 −0.5~+1.5 중립.")),
        L(B("결과 "), T("통일 레시피(C3-off) 3-seed 58.07±0.49 = published 최고 Mul-VMamba 54.65 대비 +3.42(1위). C3-on: rubber 18.80→28.56(+9.76), overall −0.10 → 예측 2/2 적중.")),
        L(B("판정 "), T("H20 ✓. 논문 헤드라인 기둥. 남은 방어 = 동일 벤치 안에서 병리 강도를 인위 조절(RailTrack 100/50/25% 서브샘플)한 인과 실험(P52.1 1-d).")),
        h2("3.4 P50: 어댑터 정렬 사전학습 (정보 축)"),
        L(B("문제 "), T("모달별 LoRA가 독립 학습되어 서로 미정렬 → 융합 트렁크가 정보를 못 씀(N2·H21에서 트렁크 타입은 비병목으로 확인).")),
        L(B("가설 "), T("Places365 200k RGB에서 pseudo-depth/lidar/event를 생성해 frozen DINOv3 위 어댑터+트렁크만 cross-modal masked reconstruction으로 사전학습하면 타깃 파인튠이 오른다. 스케일(500k·6모달)에 비례해 더 오른다.")),
        L(B("실험 "), T("tools/p50_pretrain_align.py · tools/p50_gen_pseudomodal.py. Phase1 200k×30ep vs Phase2 500k×12ep(등-스텝). 파인튠은 seed 20260821 매칭 페어.")),
        L(B("결과 "), T("Phase1 파인튠 legal test 54.95 = +0.74(게이트 +0.5 통과, 페어 1개). Phase2(500k) ep30 조기판정 53.10 = 무사전학습(53.57~54.21)보다 낮음 → 기각(user 결정 2026-09-07).")),
        L(B("판정 "), T("H22 '✓' → '재현 대기'로 강등(σ_pair 0.6 안). 스케일 역전 원인 후보 = ①pseudo-모달 = f(RGB)라 새 정보 없음 ②파인튠 초반 정점 후 망각 ③6모달 동시 사전학습의 용량 경쟁. 학습 0 진단 D1~D5(LogME·모달 Fisher·intruder-dim·CKA·WiSE-FT 보간) 예약.")),
        h2("3.5 융합 기제 소거 — 4갈래 모두 닫힘"),
        table([
            ["갈래", "실험", "결과", "판정"],
            ["선택 (패치별 센서 라우팅)", "오라클 spatial routing + no-GT 신호 3종(majority/consensus/confidence)", "GT 회수 여지 +8.5 실재하나 no-GT 신호 전부 음수(−1.9 / −2.0 / −12.5). 옳은 소수 모달은 합의로도 신뢰도로도 안 보임", "H16 ✗ (추론 시 선택 불가)"],
            ["attention 트렁크", "cross-attn 트렁크 vs gated-MLP drop-in 교체(동일 레시피)", "54.94 < 56.99 (−2.05), 파라미터 15.9×로도 패배 (구 드라이버 기준)", "H17 ✗"],
            ["산술평균 융합", "MLE-SAM식 평균 vs gated-MLP (seed821 매칭)", "평균 55.45 ≥ gated-MLP 54.2~55.4", "H21 ✗ → '우리 트렁크 우위' 주장 철회"],
            ["인코딩-시간 결합", "P51-CMLC LoRA 부분공간 결합 on/off 매칭 페어", "Δ −0.82, fog −2.03 · night −0.58 · sun +0.26", "H19 ✗"],
        ]),
        P(B("결론 "), T("노벨티 축을 '융합 기제'에서 '진단-구동 처방(H20) + 어댑터 정렬 사전학습(H22) + 소거 증명 체인 + 통일 레시피 일반성(MCubeS)'으로 재중심화.")),
        h2("3.6 MUSES: radar · 4모달 · 모달 균형 · 공정성"),
        L(B("radar "), T("3-seed race(4모달 val 82.35/82.05/81.82 전부 3모달 82.62 미달) + drop-radar ablation +0.13 + 공식 test 79.571(3모달 −0.217) → radar 무익 확정(H11). 문헌 정합: CAFuser도 lidar 위에 radar +0.6.")),
        L(B("event "), T("RGB-L 2모달 val 82.00 / 공식 test 79.571 = 3모달 −0.217 → event 기여 ≈0.")),
        L(B("LiDAR 밀도화(P47-D1) "), T("val 82.58(4모달 역대 최강) → 공식 test 78.790(−0.781). val 과적합 3례째. 폐기.")),
        L(B("모달 균형(P47-2 UniBal) "), T("모달별 독립 aux CE. val-best 82.06(게이트 82.62 미달), 공식 val 81.72. λ_u 순서가 ep0~300 불변(std 0.01~0.02) → 컨트롤러가 상수로 퇴화(modality competition 이론과 일치).")),
        L(B("공정성 정정(2026-09-07, user) "), T("MUSES 최선 레시피가 PhysAug(물리 열화 증강) on이었음. 조사 논문 전부 표준 증강만 사용 → 카드 E7로 PhysAug-off 기준선 재런, 완주 후 E7c(on)와 공식 채점기로 페어 비교해 헤드라인 교체 여부 결정.")),
        IMG("fig3_muses_leaderboard.png", "MUSES 공식 test 리더보드. 주황 = 반증된 시도(radar 추가·밀도화·BMR·C3 이식)."),
        h2("3.7 P52 RxDINO와 타당성 감사"),
        L(B("설계 "), T("단일 config 자기-적응: C3-adaptive(train 혼동 EMA → 클래스 붕괴 점수 → per-class λ) + UniBal-adaptive(모달별 aux 손실갭 → per-modal λ) + P50 Phase1 init. 추론 그래프는 3벤치 동일. 구현 semseg/models/reliadino/p52.py, 기본 off = byte-동일.")),
        L(B("감사 판정(2026-09-07) "), T("①게이트 '고정팔 −0.3 이내' = 시드 σ(0.76~0.92) 아래라 판별 불가 ②C3 신호가 train 혼동인데 고치려는 병리는 val→test 붕괴(train recall 높은 RailTrack에서 λ가 안 켜질 수 있음) ③UniBal 신호 ep0 불변 = 고정 가중과 동치 ④선행 Recall Loss·CGRm / PMR·MLB가 각 기둥 점유 ⑤'config 차이 0' 주장은 실물 config(P50 init·PhysAug·해상도·모달 수 4건 차이)가 부정.")),
        L(B("수정(P52.1) "), T("신호 즉검(진행 런 [C3-ADPT] 로그) → 게이트를 동등성 밴드 [−1.0, ∞) 3페어 + 고정 가중 스윕 −0.5 이내(G5) + λ 궤적 제출로 재등록 → 병리 조작 실험 → 주장 문구 축소 → MUSES 3모달 기준선 병기.")),
        L(B("본런 현황 "), T("MCubeS seed1 완주 58.18@ep174(final 57.96) = 3-seed 정본 58.07±0.49 대역 안 → 동률. DELIVER seed1/seed2 ep38 val 66.25/66.19(yeon). MUSES seed1/seed2 hpca100 보류(ep54 79.63 / ep30 78.81, ckpt 보존).")),
        h2("3.8 베이스라인 직접 재학습 (2026-09-08 착수)"),
        L(B("목적 "), T("DGFusion(depth-guided fusion)의 재현 수치를 통계적으로 확보한 뒤 한계를 분석해 우리 설계에 반영. 공개 저장소에 학습 코드가 의도적으로 빠져 있어(train_net.py 부재, 학습 분기 차단) CAFuser 공개 코드를 대조해 복원.")),
        L(B("실행 "), T("DGFusion Swin-T DELIVER CLDE 공식 설정(bs8·LR 1e-4·200k iter) jarvis GPU1,2,4,5, ETA ~1.3일 · CAFuser Swin-T 대조군 lecun GPU0,1,2(3 GPU 제약: bs6·LR 0.75e-4·266,667 iter = 총 샘플 동일). 복원 킷 third_party/dgfusion_train_restore/.")),
        L(B("기대 산출 "), T("공개값(DGFusion val 66.51/test 56.71, CAFuser 68.12/55.80) 재현 여부 + per-class 3자 비교표(우리 P46 5-seed / DGFusion 재현 / MM SAM-adapter)의 재료.")),
        IMG("fig2_deliver_campaign.png", "DELIVER 캠페인 합법 test 정리. 초록 = 게이트 통과, 주황 = 반증, 노랑 = 동급(우위 주장 철회)."),
    ]
    return out

# ───────────────────────────────────────────── 4. 일일 카드
def sec_cards():
    return [
        P(B("user의 문제 정의 "), T("DELIVER는 분할 형상은 괜찮은데 클래스를 헷갈려 점수를 잃는다. 센서마다 클래스 근거가 다르므로 어댑터가 그것을 배우게 하면 오를 것이다. MUSES는 압도적 SOTA가 아니다. 두 벤치에 공통으로 한 단계 올리는 방법이 필요하다.")),
        h2("4.1 공통 스크린 규약"),
        table([
            ["항목", "값"],
            ["레시피", "DELIVER 4센서(img/depth/event/lidar), P46 C3-only λ0.1, PhysAug off, DGFUSION_AUG on, 768² 학습"],
            ["시드 · 길이", "20260821 매칭(모든 카드 동일) · 40 epoch, EVAL_INTERVAL 5"],
            ["판정 수치", "40ep 이내 legal-val 최고 ckpt를 val.py native-GT BS1(하네스 가드)로 test 재채점"],
            ["기준선 B0", "같은 규약의 현행 레시피 40ep = legal test 53.78 / val 64.97 (트레이너 val 65.4, 정본 대비 +0.43)"],
            ["스크린 통과", "Δtest(카드 − B0) ≥ +1.0 (= test ≥ 54.78) and 악조건(night·fog·rain) 어느 것도 −0.5 미만 아님"],
            ["확정", "통과 카드만 200ep × 3페어, 게이트 = 3페어 mean ≥ +1.0"],
            ["조기 kill", "ep20 legal-val이 B0 ep20 −1.5 미만"],
            ["구현 규칙", "카드마다 config 토글 하나, 기본 off에서 forward byte-동일(스모크: state_dict 키 + |Δ|max=0)"],
        ]),
        h2("4.2 카드 표 (2026-09-08 16:10 KST 기준)"),
        table([
            ["카드", "겨냥한 문제", "가설", "변경(1개)", "상태", "결과", "판정"],
            ["B0 기준선", "—", "—", "없음 (EPOCHS 40)", "완주 (bengio GPU0-3)", "legal test 53.78 / val 64.97", "대조군 고정"],
            ["E0 특징 정보 프로브", "어댑터가 백본 정보를 버리는가", "클래스·센서 정보가 DINOv3 원 특징에는 있는데 어댑터·헤드가 버린다", "학습 = 선형 헤드만 (tools/probe_feature_info.py)", "완료", "선형 프로브 test mIoU: 원 특징 27.8 < 어댑터 후 35.6 < 융합 45.1. depth 중간층 탭에 Water 47.6·RailTrack 23.9 잔존(융합 16.6·0.0)", "전역 가설 기각 → E5·E6 하향, E1·E3 유지"],
            ["E9 로짓 사전 보정", "클래스 사전의 도메인 이동", "추론 시 logit − τ·log p_val", "val.py --logit-adjust-tau", "완료", "τ=0/0.5/1.0 → 53.57/53.45/51.89 (mAcc는 상승)", "폐기 (mIoU 단조 하락)"],
            ["E1 중간층 4탭 읽기", "기하 센서의 클래스 근거가 중간층에 있음", "블록 6/12/18/24 출력을 SimpleFPN 레벨별로 투입하면 depth·lidar 의존 클래스가 회복", "MODEL.TAPS per_modal [6,12,18,24]", "완주 + legal test 재채점 완결 (bengio), legal val 재채점 중", "트레이너 val ep5~35: 60.98/63.56/66.30/66.57/66.43/66.88/67.25(카드 중 최고) vs B0 58.71/63.17/62.18/63.91/64.03/65.13/64.67(ep40 65.40)", "✅ 스크린 통과 — legal test 54.85 = B0 +1.07 ≥ +1.0 (val-best ep35, 생각정리 세션 확정). 클래스별: RailTrack +13.13·TrafficLight +5.01·SideWalk +3.09, Wall +0.26·Static +0.25·Water −1.71. 다음 = 200ep×3페어 확정 런. val 순위(E2>E1)와 test 순위(E1>E2) 역전"],
            ["E2 전 선형층 LoRA", "Q/V만으로는 센서별 클래스 근거를 담을 용량이 없음", "LoRA를 Q/K/V/O + MLP fc1/fc2, r32·α64로 확장", "LORA_TARGETS [qkv_full, proj, fc1, fc2]", "완주 + legal 재채점 완결 (hpca100)", "legal test 54.50 / val 68.38 (val-best ep40, 1024 BS1). B0 대비 Δtest +0.72 / Δval +3.41. 클래스별: RailTrack +15.77·TrafficLight +12.49 / Wall −6.40·Water −5.47·Static −5.44. 트레이너 val 68.65 − 공식 68.38 = +0.27", "판정 보류 — Δtest +0.72는 통과선 +1.0과 폐기선 +0.5 사이(회색지대). 판정 = 생각정리 세션"],
            ["E3 센서별 클래스 prototype", "user 가설 직접 검증", "클래스 정체성을 센서마다 독립으로 지지하면 RGB가 헷갈리는 클래스를 lidar·depth 근거가 붙잡음", "P46.C3_PROTO.SRC permodal + AGREE_LAMBDA", "완주 (bengio GPU4,5, 20:53:16)", "ep5~40: 59.35/63.72/62.23/65.44/65.97/65.51/65.61/65.69 (val-best 65.97@25). 로그 Best Test 54.99@35는 test-best라 무효", "보류 — legal 재채점 대기열"],
            ["E4 혼동 쌍 margin", "붕괴가 특정 쌍으로 흡수됨", "검증셋 혼동행렬 top-k 쌍에 prototype/로짓 margin", "CONFUSION_PAIRS auto_k5", "ep30/40 (bengio GPU1-3), 완주 ETA 22:30 KST", "ep5~25: 59.77/63.17/65.92/65.70/65.79 (best 65.92@15)", "보류. auto 쌍에 RailTrack 누락 → 명시 쌍판 E4b config 추가(RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain), 미기동"],
            ["E7 MUSES PhysAug-off", "공정선 정합", "헤드라인 교체용 기준선(게이트 없음)", "PHYSAUG off", "완주 (hpca100 GPU2)", "트레이너 val ep5~40: 73.84/75.39/77.06/77.41/79.16/79.84/79.69/80.29", "E7c와 페어 공식 재채점 진행 중(첫 공식 test)"],
            ["E7c MUSES PhysAug-on 대조군", "PhysAug 효과 크기 격리", "—", "E7과 PHYSAUG만 상이", "완주 20:25 KST (hpca100 GPU2)", "ep5~40: 73.18/75.52/75.97/77.68/79.54/79.52/79.96/80.18. E7 대비 8지점 Δ −0.66/+0.13/−1.09/+0.27/+0.38/−0.32/+0.27/−0.11", "보류 — 트레이너 val 차이 ±1.1 안·부호 교대, 공식 재채점 결과로 판정"],
            ["E1M MUSES 4탭 이식", "E1의 MUSES 공통 상승 검증", "중간층 4탭이 MUSES에서도 오르면 공통 한 단계", "E7(PhysAug-off) + MODEL.TAPS on", "ep18/40 (hpca100 GPU3, 디스크 만복으로 중단 후 /tmp 우회 재개), ETA 09-09 18시 KST", "ep5 71.73 · ep10 75.55 vs E7 73.84 · 75.39 (−2.11 → +0.16, warmup 15ep라 ep15·20이 실질 비교점)", "E7 매칭 페어, 공식 val Δ ≥ +1.0"],
            ["E12 / E2s2 (hpca100)", "E1+E2 결합 / E2 시드2 페어", "결합이 E1·E2 이득을 더하는가 / E2 +0.72의 재현", "TAPS on + LORA_TARGETS 전 선형층 / E2 + seed 20260902", "ep2/40 · ep1/40 (디스크 만복으로 /tmp 우회 가동)", "—", "09-10 08시 · 07시 KST 완주 예정"],
            ["B0s2 / E1s2 / E4b (bengio)", "시드2 기준선 / E1 시드2 페어 / RailTrack 명시 쌍 margin", "E1 통과의 재현 · 명시 쌍이 RailTrack을 더 올리는가", "seed 20260902 / E1 + seed 20260902 / PAIRS 명시", "ep3/40 · 기동 중 · ep5/40", "—", "Δtest ≥ +1.0 (E4b는 + RailTrack > B0 31.98)"],
            ["E8 / E5 / E6 / E10", "표적 copy-paste / deformable 픽셀 디코더 / 매 블록 교환 어댑터 / 상위 절반 FT", "—", "구현 2~5일", "미착수 (E5·E6은 E0 결과로 하향)", "—", "—"],
        ]),
        IMG("fig7_daily_cards.png", "DELIVER 카드의 트레이너 val 궤적(판정용 아님)과 사전 등록 판정 기준."),
        IMG("fig4_e0_probe.png", "E0 특징 정보 프로브: (a) 정보량은 raw < adapted < fused, (b) 붕괴 클래스 근거는 depth 중간층에 남고 융합이 잃는다."),
        h2("4.3 판정이 다음 카드로 이어진 고리"),
        L("E0 '어댑터가 버린다' 기각 → 어댑터 확대 계열(E5 deformable 디코더·E6 블록 교환) 하향, 중간층 읽기(E1)·센서별 prototype(E3) 우선."),
        L("E4 auto 쌍에 RailTrack이 빠짐 → 명시 쌍판 E4b 신설."),
        L("E1 궤적 양성 → 시드2 매칭 페어(B0s2·E1s2) + MUSES 이식(E1M) + E2와 결합(E12) config 준비."),
        L("E9 폐기 → 혼동은 클래스 사전 이동이 아니라 재현율↔정밀도 교환으로만 반응 → 손실·특징 접점 카드로 집중."),
    ]

# ───────────────────────────────────────────── 5. Related works
def sec_related():
    def tbl(rows): return table([["방법", "출처", "핵심", "우리와의 관계·차별점", "인용"]] + rows)
    return [
        P("관련연구는 '우리가 어느 칸에 서 있고, 누구와 구분해야 하는가' 기준으로 묶는다. 인용은 arXiv id 또는 venue. ★ = 반드시 인용, ○ = 비교표 행, △ = 정독·확인 후 결정."),
        h2("5.1 벤치마크 경쟁작 (DELIVER · MUSES · MCubeS)"),
        tbl([
            ["CMNeXt", "CVPR 2023, 2303.01480", "DELIVER 벤치 제안, Self-Query Hub로 임의 모달 융합, MiT-B2", "val 66.30(B2) / test 53.0(CLDE). 프로토콜 3클러스터 주의(59.18은 B0)", "★"],
            ["CAFuser", "RA-L 2025, 2410.10791", "CLIP 텍스트 조건 토큰으로 Condition-Aware Cross-Attention(CA²)/Addition(CAA)", "val 67.8/68.6, test 55.6/55.2 · MUSES 78.5. 외부 텍스트 신호 의존. 우리 대조군으로 lecun에서 재학습 중", "★"],
            ["DGFusion", "RA-L 2026, 2509.09828", "depth(입력+GT)로 spatially-varying sensor reliability → depth token이 융합을 condition", "test 56.7(리사이즈 GT) · MUSES 79.5. 최근접 경쟁자. 학습 코드 복원 후 jarvis에서 공식 설정 재학습 중", "★"],
            ["MM SAM-adapter", "IEEE 2025, 2509.10408", "SAM ViT-L + 비RGB 별도 인코더 + 비대칭 주입 어댑터", "DELIVER test 57.35(RGB-D, native GT = 우리와 동일 자) = 현 SOTA. frozen 55.35→FT 57.14. 우리 P49 이식 실패(−0.8~−0.97)", "★"],
            ["StitchFusion", "ACM MM 2025, 2408.01343", "frozen 대형 백본 + 매 블록 MoA 교환 어댑터", "val 68.18(val 상위). 우리 P51 결합(−0.82)과 정신 유사 → 차이(블록 출력 교환 vs LoRA 코드 결합)를 먼저 써야 함(E6 전제)", "★"],
            ["GeminiFusion", "2406.01210", "블록 attn-MLP 사이 픽셀단위 cross-attention 교환", "test 54.5. 강건성 벤치에서 노이즈 조건 급락 = 교환량↔강건성 트레이드오프 사례", "○"],
            ["OmniSegmentor", "2509.15096", "ImageNeXt 정렬 사전학습(+2.4~5.1), 동시 다모달 사전학습은 −0.2~−0.5·교대만 +", "val 68.0(DFormer-L). 우리 P50 스케일업 실패 원인 후보 3(용량 경쟁)의 근거", "★"],
            ["MAGIC / HyperDUM / AnySeg", "2407.11344 / CVPR 2025 / 2411.17141", "modality-agnostic · hyperdimensional uncertainty · uni/cross-modal distillation", "val 67.66 / 67.59 / —. 신뢰도·결측 축 이웃", "○"],
            ["Mul-VMamba · MMSFormer", "KBS / IEEE OJSP 2024, 2309.04001", "MCubeS published 최고 54.65 / 53.11", "우리 58.07±0.49 = +3.42", "★"],
            ["MUSES 데이터셋 · GtA", "2401.12761 / Codabench", "MUSES 벤치 제안 · 리더보드 1위 82.39(camera-only)", "GtA는 1차 출처 미확인(로그인 장벽) → 제출 전 재확인 의무", "★ / △"],
        ]),
        h2("5.2 신뢰도·품질·조건 가중 융합 (우리가 반증한 축 — 구분 인용)"),
        tbl([
            ["MemorySAM", "2503.06700 (preprint 유지)", "SAM2 memory attention을 모달 축으로 전용, 신뢰도 개념 없음", "우리 토대. venue 추적 유지(학회 표기 1차 출처 없음)", "★"],
            ["DFormerv2", "CVPR 2025, 2504.04701", "depth 기하 prior로 self-attention 가중 변조(GSA)", "'보조 모달을 attention 변조로 주입' 축의 최근접 CVPR 이웃. additive/multiplicative 정독 후 비교표 행 추가", "★ (△ 정독)"],
            ["RSGMamba / EGFormer", "2604.12319 / 2505.14014", "self-gating(곱셈) 신뢰도 · 모달 중요도로 저정보 모달 탈락", "clean 이득은 소형 백본에서만 ≤1.5, Swin-T급 이상 0~0.4 — 우리 30세대 실패와 정합", "○"],
            ["UTFNet / ReliFusion / READ", "GRSL 2023 / 2502.01856 / ICLR 2024", "evidential head · 학습 reliability 출력 스케일 · loss 가중 TTA", "feature/output/loss 레벨 vs (우리 구 RBMA) logit-additive. 성능 근거 없어 브랜딩 내림", "○"],
            ["PRIMED / SAE", "2605.07154 / 2603.16558", "learned modality-prior additive pre-softmax bias / training-free entropy additive bias(LVLM)", "RBMA 셀의 최근접 위협 — 전문 정독 TODO(blocking)", "△"],
            ["CoRiM / UMFNet / ACR", "CVPR 2026", "Modality Conflict Risk 최소화(scalar confidence 한계 이론) / 픽셀 Gaussian 불확실성 / confidence degradation 벌점", "신뢰도 가중 계열 포지셔닝 시 이론 인용", "○"],
            ["RAF / InfraNet / GIML", "ECCV 2026 2607.04587 / 2607.03795 / 2607.06943", "per-pixel 학습형 reliability map / 품질 게이트 / 결손·열화 연속 품질 추정", "'품질로 불량 모달 억제' 스토리는 혼잡한 채택 클러스터 → 메커니즘 차별화 전면", "○"],
            ["센서 결측·노이즈 벤치", "CVPRW 2025 Best Paper, 2503.18445", "DELIVER EMM/RMM/NM 프로토콜", "이후 DELIVER 신작 대부분 결측 수치 병기 → 채택 여부 결정 필요(리뷰어 요구 가능성)", "★"],
            ["RobustSeg / MARS / Missing No More", "CVPR 2026 / 2606.30355 / 2603.08018", "결측 모달 증류·MoE·pseudo-IR 추론", "결측 실험 시 직접 비교 대상", "○"],
        ]),
        h2("5.3 파운데이션 백본 적응 (frozen vs LoRA vs FT, 어댑터 깊이)"),
        tbl([
            ["DINOv3 / DINOv2", "2508.10104 / 2304.07193", "frozen 특징 권장(배포 효율 근거), 4탭 이득은 depth 등 기하 모달에서 +2~4", "우리 백본. E1(4탭) 근거", "★"],
            ["Rein", "2312.04265", "DINOv2-L Freeze 61.1 / Full FT 61.7 / LoRA 62.7 / Rein 64.3, Full FT 역U 과적합", "frozen 특징이 좋을수록 FT 이득 작음 → E10 부분 FT 게이트 설계 근거", "★"],
            ["LoRA Learns Less and Forgets Less / SoMA", "2405.09673 / 2412.04077", "MLP가 학습 주 loci, attention-only < MLP ≈ all, α=2r · attn+MLP LoRA +1.5", "E2(전 선형층 LoRA r32·α64) 근거", "★"],
            ["SHIFNet / SpectraDINO / DPLNet", "IROS 2025 2503.02581 / 2605.02258 / 2312.00360", "매 블록 어댑터(Q/V만 쓴 SARTM 대비 +6.2) / 상위 절반 FT +1.4~1.8, 전체 FT −9.4 / 프롬프트", "5개 벤치 1위 어느 것도 'Q/V LoRA + 마지막 층 + 후기 융합'이 아님 → 구조 사다리 S1~S4", "○"],
            ["LP-FT / surgical FT / LLRD", "2202.10054 / 2210.11466 / ViTDet 2203.16527", "선형 프로브 후 FT · 층 선택 FT · 층별 lr 감쇠", "E10 처방 세부", "○"],
            ["M4-SAM / SENTRY / KAN-SAM / SAM3-Adapter / REALM", "CVPR 2026 2605.11760 / ECCV 2026 2606.24449 / 2504.05878 / 2511.19425 / 2605.00271", "SAM2+MoE-LoRA(memory는 시간축) / training-free SAM2 memory 개입 / thermal KAN 프롬프트 / SAM3 어댑터 / event→RGB 파운데이션 latent 투영", "SAM 계열 구분 인용. 비RGB 센서 SAM 적응은 양대 학회 0건 = 선점 여지", "○"],
        ]),
        h2("5.4 클래스 불균형 · 모달 균형 손실 (P46-C3 · P52 선행)"),
        tbl([
            ["Recall Loss / CGRm / Seesaw / LDAM-DRW", "2106.14917 / 2608.09688 / 2008.10032 / 1906.07413", "recall·혼동행렬 기반 per-class 가중", "P52 C3-adaptive의 최근접 선행. 남는 차별 = recall×혼동집중도를 prototype-contrastive 항에 적용", "★"],
            ["PMR / MLB / OGM-GE / G-Blending / MMPareto", "2211.07089 / 2405.07930 / 2203.15332 / 1905.12681 / 2405.17730", "느린 모달 가중·보조 CE 균형", "UniBal-adaptive와 사실상 동형(PMR). MMPareto: uniform 보조 CE ≥ OGM-GE/PMR", "★"],
            ["Modality competition", "2203.12221", "승자 모달이 초기화 단계에서 결정되어 유지", "우리 λ_u ep0 불변 관찰과 정확히 일치", "★"],
            ["EQUISeg / 함수 엔트로피 정규화", "2509.24505 / ICCV 2025 2505.06635", "고정 λ 균형법 / unimodal bias 감소", "'적응형 > 고정형'을 직접 보여야 함. 함수 엔트로피는 결측 강건성용이라 우리 목표와 불일치 → 폐기(2026-09-07)", "○"],
            ["Xin et al. / Gama & Grassi / BalanceBenchmark", "2209.11379 / 2505.10347 / 2502.10816", "적응 가중이 거의 안 움직임·고정 가중과 동급·절대 균형 추구 시 baseline 이하", "P52 리뷰어 반론의 근거 — 궤적 플롯·고정 스윕 대비 의무", "○"],
        ]),
        h2("5.5 사전학습 · 정렬 · 망각 (P50)"),
        tbl([
            ["MultiMAE / DFormer / Zoph et al.", "2204.01678 / 2309.09668 / 2006.06882", "pseudo-모달 < 실센서 · depth 사전학습 +15.2 · task-alignment 부재 시 사전학습 −1.0", "P50 스케일 역전 원인 후보 1(pseudo-모달 = f(RGB))", "★"],
            ["Andreassen / intruder dims / LogME / CKA", "2106.15831 / 2410.21228 / 2102.11005 / 1905.00414", "중간 정점 후 소멸 · LoRA 새 특이벡터와 망각 ρ=0.97 · init 적합도 · 특징 이동", "학습 0 진단 D1~D5 도구", "○"],
            ["WiSE-FT / L2-SP", "2109.01903 / 1802.01483", "가중치 보간 · 사전학습 기준점 weight decay", "원인 2(망각) 처방 후보", "○"],
            ["Bouthillier et al.", "2103.03098", "시드 노이즈 ≈1.5pt, 초기화만 바꾸면 분산 과소추정", "축 0 측정 규약(R4) 근거. DELIVER/MUSES SOTA 논문 중 시드 std 보고 0건", "★"],
        ]),
        h2("5.6 검출 트랙 (RGB-IR/LiDAR 융합 검출)"),
        tbl([
            ["OAFA / CoDAF / JFRDet / DPDETR", "CVPR 2024 / 2506.16737 / 2608.10680 / 2408.06123", "미스얼라인 3세대: offset 명시 추정 → affine+deformable → 모달별 박스 출력", "poongsan RGB-T-L 정합 논의 시 인용", "○"],
            ["MDQF / DAMSDet / MS-DETR", "2601.08458 / ECCV 2024 / TITS 2024 2302.00290", "모달별 DETR 브랜치 + 고품질 쿼리 선별, 열화 모달 배제", "'모달별 표현 품질 확보 후 선택적 융합' 흐름", "○"],
            ["M²D-LIF", "2503.11780", "joint 멀티모달 학습이 단일모달 표현을 부실화(Fusion Degradation, linear probing으로 정량)", "poongsan 'RGB-only ≥ 3모달' 관찰의 원인 규명 도구로 적용 예정", "★"],
            ["AMFD / FreqKD", "TMM 2025 2405.12944 / 2606.11572", "융합-전 원 모달 특징 증류 / 대역별 비대칭 증류(uniform KD 전부 baseline 미달)", "GISTOLO(D1 ViT-S+ → YOLOv5m RGB 증류)와 직접 비교", "○"],
            ["Thermal-Det / UniRGB-IR / YOLOv11-RGBT", "CVPR 2026 2605.10130 / ACM MM 2025 2404.17360 / 2506.14696", "OV 검출기 thermal 이식 / frozen RGB 파운데이션에 IR 어댑터 / 6융합모드 통일 프레임워크", "인증 모델·베이스라인 인프라 인용", "○"],
            ["SegFly / MMVIP / DSERT-RoLL", "ECCV 2026 2603.17920 / CVPR 2026 / 2604.03685", "항공 RGB-T 세그 데이터셋 / 해양 VIS-IR 128K 페어 / stereo event+RGB+thermal+radar+LiDAR 주행 데이터셋", "드론·해양(MULTIAQUA) 도메인 인접 — 데이터셋 절 인용", "○"],
        ]),
        h2("5.7 채워야 할 항목 (정독·확인 TODO)"),
        L("DFormerv2 GSA 수식 정독 → attention 변조가 additive / multiplicative / 거리감쇠 중 무엇인지 확정 후 §5.2 행 확정."),
        L("PRIMED(2605.07154)·SAE(2603.16558) 전문 정독 — 'training-free 신뢰도 → pre-softmax additive bias' 셀의 점유 여부(blocking)."),
        L("GtA 82.39(camera-only)의 1차 출처 확인(Codabench 로그인) — 제출·논문 표 작성 전 필수."),
        L("MemorySAM venue 추적(현재 preprint) · MemorySAM 65.38의 split 확정."),
        L("결측·노이즈 프로토콜(2503.18445) 채택 결정 + val 루프 구현(EMM 15조합·RMM r 3단계·NM S&P)."),
        L("프리프린트 미공개 채택작 추적: RA-SOD, 'DETR is Secretly a Multispectral Detector'(zero-parameter), BIP, Dark-Scenes depth, MMVIP, CoRiM, DyFCLT."),
        L("MUSES panoptic 챌린지 리포트(URVIS 워크숍, 2604.16984) 정독 — MUSES 동향."),
    ]

# ───────────────────────────────────────────── 6. 한 것 / 할 것
def sec_plan():
    return [
        P(B(f"레포 실험 계획 문서(experiments/plan)와 동기화 ({TODAY}). "), T("실시간 상태는 레포가 정본, 이 절은 같은 날짜 스냅샷.")),
        h2("6.1 진행 중 (서버별)"),
        table([
            ["서버 / GPU", "실험", "데이터셋", "진행", "ETA · 비고"],
            ["bengio 0-3", "B0 기준선 스크린", "DELIVER 4모달", "완주 → legal test 53.78 / val 64.97", "카드 대조군 고정"],
            ["bengio 6,7 → E1s2 기동", "E1 중간층 4탭 읽기 → 시드2 페어 E1s2", "DELIVER 4모달", "E1 완주, legal test 54.85 (B0 +1.07, 스크린 통과) · E1s2 기동 중", "E1 legal val 재채점 중(GPU0), 이어서 E4 재채점"],
            ["bengio 4,5 → B0s2", "E3 센서별 prototype 완주 → 시드2 기준선 B0s2", "DELIVER 4모달", "E3 val-best 65.97@25(legal 대기) · B0s2 ep3/40", "E3 legal 재채점 대기열"],
            ["bengio 1-3 → E4b", "E4 혼동 쌍 margin(auto) 완주 → 명시 쌍판 E4b", "DELIVER 4모달", "E4 val-best 65.92@15(legal 대기) · E4b ep5/40", "E4 legal 재채점은 E1 val 다음"],
            ["hpca100 1,3 → E12 · E2s2", "E2 완주(legal 54.50/68.38, Δtest +0.72 회색지대) → E12(E1+E2 결합) ep2/40 · E2s2(시드2) ep1/40", "DELIVER 4모달", "디스크 만복으로 /tmp 우회 가동", "09-10 08시 · 07시 KST 완주 예정"],
            ["hpca100 3", "E1M MUSES 4탭 이식(E7 매칭 페어)", "MUSES 3모달", "ep18/40 재개(디스크 만복 중단 후), 75.55@10 (E7 75.39)", "09-09 18시 KST 완주 예정"],
            ["hpca100 2", "E7·E7c 공식 페어 재채점(tools/eval_muses_official.py)", "MUSES 3모달", "E7c 완주 80.18@40 / E7 80.29@40, 재채점 진행 중", "E7의 첫 공식 test 수치 → PhysAug 효과 확정"],
            ["yeon 2,3 / 4,5", "P52 RxDINO DELIVER seed1 / seed2", "DELIVER 4모달", "ep38/200, val 66.25 / 66.19", "게이트 G1 54.65(재등록 예정)"],
            ["yeon 0,1", "E-LoRA arm A (per-modal r16 재런)", "DELIVER 4모달", "ep38/200, val 66.64@26", "arm B(공유)·C(공유+잔차) 슬롯 대기"],
            ["hpca100 0,2 / 1,3", "P52 MUSES seed1 / seed2", "MUSES 4모달", "보류(ep54 79.63 / ep30 78.81, ckpt 보존)", "감사 §1.5 기준선 문제 적용 후 재개"],
            ["jarvis 1,2,4,5", "DGFusion Swin-T 재학습(공식 설정 bs8·LR 1e-4·200k)", "DELIVER CLDE", "iter 19,979/200,000 · iter 10k 평가 mIoU 60.31 (depth abs_rel 10.99)", "ETA ~1.3일 (2026-09-09~10). 공개값 val 66.51/test 56.71"],
            ["lecun 0,1,2", "CAFuser Swin-T 대조군(bs6·LR 0.75e-4·266,667 iter)", "DELIVER CLDE", "iter 16,339/266,667 · iter 10k 평가 mIoU 55.74", "로그 ETA 4일 16시간 (2026-09-13경). 공개값 val 68.12/test 55.80"],
            ["NAS 보존", "정본 ckpt(E2·E7·E7c val-best) md5 대조 후 /drone_nas/…/ckpts/daily_cards_20260908/ 보관", "—", "완료", "E1 ep35 ckpt도 보존 대상"],
        ]),
        h2("6.2 완료·판정 (최근, 재실행 금지)"),
        table([
            ["항목", "결과", "판정"],
            ["P52 MCubeS seed1", "val-best 58.18@174 / final 57.96", "3-seed 정본 대역 안 = 동률, 이득 없음"],
            ["N7 VICReg-off 격리", "val-best 66.56@32 (test 56.31@36은 test-best라 인용 금지)", "seed821 대비 중간 궤적 동대역 → DELIVER에서 VICReg 순기여 ≈0 가능성, val-best 시점 test 재채점 필요"],
            ["P50-EXT Phase2 채택 게이트", "ep30 legal test 53.10 (게이트 55.25)", "기각 — P52 init = Phase1 프로브 확정"],
            ["E0 / E9 / E7", "§4 카드 표", "E0 기각·E9 폐기·E7 완주(재채점 대기)"],
            ["P47-2 UniBal 고정런", "val-best 82.06@164, 공식 81.72", "게이트 82.62 미달, G2=81.42 확정, λ_u 상수 퇴화"],
            ["N1·N2·N4·N4b·N6·N8", "MUSES 시드 spread 0.66 · 평균 융합 55.45 · MCubeS 58.07±0.49 · dose-response 2/2 · DELIVER 54.39±0.76 · P50 54.95 최종", "전부 원장 반영"],
        ]),
        h2("6.3 대기열 (우선순위 순)"),
        table([
            ["#", "실험", "왜", "자원", "게이트"],
            ["1", "✅ E1 스크린 통과(legal test 54.85 = +1.07) → 200ep × 3페어 확정 런 착수", "카드 프로그램 첫 통과. 시드2 페어(E1s2·B0s2) 이미 기동", "yeon 2장 × 3 (P52·E-LoRA 완주 09-12~13 후 슬롯)", "3페어 mean ≥ +1.0"],
            ["2", "E2 회색지대(+0.72) 재판정 = E2s2·E12 결과로 / E3·E4 legal 재채점 / E4b 스크린", "2~3일차 판정 완결", "bengio GPU0 재채점 큐 · hpca100", "Δtest ≥ +1.0 (페어 mean)"],
            ["3", "E7 vs E7c 공식 재채점(tools/eval_muses_official.py) → MUSES 헤드라인 교체 여부", "공정성 정정", "1 GPU 수 시간", "PhysAug 효과 크기를 증강 ablation으로 공개"],
            ["4", "E1M(MUSES 4탭) · E12(E1+E2) · B0s2/E1s2 시드2 페어", "공통 상승·결합·재현", "hpca100/bengio", "양 벤치 +1.0 = '공통 한 단계'"],
            ["5", "클래스별 3자 비교표(우리 P46 5-seed / DGFusion 재현 / MM SAM-adapter, 학습 0) · 오라클 지도 라우터 프로브(H26)", "'강점 유지·혼동 개선' 실증 · 학습형 라우팅 영구 폐쇄 여부", "GPU 1장 재채점", "라우터 val 정확도 − 우연 ≥ +10%p"],
            ["6", "P52.1 재설계: 신호 즉검([C3-ADPT] 로그) → 게이트 재등록(동등성 [−1.0,∞) 3페어 + 고정 스윕 −0.5 이내) → 병리 조작(RailTrack 100/50/25%)", "감사 6항목", "시드 +1 3런 + 3런", "G1~G5"],
            ["7", "P50 Phase1 재현(+2페어) → 학습 0 진단 D1~D5 → P50.1 처방 1개", "H22가 페어 1개 결과", "yeon 2장 × 4런", "3페어 mean δ ≥ +0.7"],
            ["8", "E-BB A1(= E2 확정) → A2 상위 절반 FT → E8 copy-paste → E5/E6(하향)", "구조 사다리 S2~S4", "yeon 2장 × 3런씩", "A1 ≥ A0 + 1.0 / A2 ≥ A1 + 0.5"],
            ["9", "N9 정성 증명 패키지(rank 스펙트럼·confusion 전후·per-class 예측맵) · N5 TTA ablation", "논문 그림", "1 GPU 간헐", "—"],
        ]),
        h2("6.4 종결 (재제안 금지)"),
        L("attention-bias 계열(RBMA) · 추론 재가중(gate/calib/veto) · CEFR · zero-init 잔차(5회 사망) · rank/η² 개입 · 모달 드롭 · gradient 균형화 · radar(MUSES) · NORM_ALL · P48 인스턴스 감독(PQ 22.87) · P40 RCA · P49 비대칭 주입 · cross-attn 트렁크 · 인코딩-시간 결합(P51) · 함수 엔트로피 정규화(원문 확인 후 폐기) · 백본 L→7B."),
        h2("6.5 논문 트랙"),
        L("분기: CVPR 2027(마감 2026-11 중순) vs RA-L. 판정 시점 = 카드 S1(E1)·S2(E2) 3페어 완주(약 3주). DELIVER 정본 test ≥ 57.35 또는 MUSES 공식 val ≥ 82.13(PhysAug off)을 넘으면 방법 논문, 못 넘으면 분석 논문(소거 증명 + dose-response + 통일 레시피)."),
        L("기여 문구 재정의 대기(user 결정): (A) 진단 지표로 λ 초기화 규칙 / (B) 소거 증명 + dose-response + 통일 레시피 본론 / (C) 컨트롤러 주장 삭제. 권고 = B 본론 + A 보조."),
        L("RA-L 초안(ReliaDINO v1)은 RBMA 중심 서사라 재중심화 필요."),
    ]

# ───────────────────────────────────────────── 7. 검출 트랙
def sec_det():
    return [
        table([
            ["항목", "결과", "비고"],
            ["목표", "poongsan indoor mAP50 0.85 (국가 R&D)", "2026-07-04 달성"],
            ["최선", "D1-recovered(ViT-L) AP50 0.9321@ep6", "egofill LiDAR 데이터(2.01×)만으로 0.8501@ep9 선행 달성"],
            ["인증 모델", "D1 ViT-S+ 3모달, RTX 5090 19.43 fps, 야간 mAP50 0.8292", "branch 26-drone-certificate, GPU 미인식 원인 = CPU-only torch(cu128 재설치로 해결)"],
            ["모달 ablation", "final annotation: RGB-only 0.7964 ≥ 3모달 0.7895 (mAP50)", "egofill lidar+thermal은 strict-IoU mAP에만 도움 → 'RGB-only ≥ 3모달' = Fusion Degradation(M²D-LIF) 진단 적용 예정"],
            ["증류(GISTOLO)", "D1 ViT-S+ → YOLOv5m RGB, 3/3 seed 재현 우위 +0.005(작음)", "0.9166(egofill 조밀 3239) vs 0.9081(raw 2066)은 데이터 차이"],
            ["스택 진단 이력", "P29-Det 0.446 → 라벨 수리(v2) → egofill 0.8501; P30-Det 소물체 붕괴(AP_small 0.006) → 폐기", "YOLO11m RGB-only 0.864 = '데이터 무죄, 스택 유죄' 판정 근거"],
        ]),
        P("남은 검출 스토리 = 저조도(야간) delta 실증 + 융합이 단일모달 표현을 부실화시키는지(linear probing) 정량화."),
    ]

# ───────────────────────────────────────────── 8. 규약
def sec_rules():
    return [
        L("ckpt는 val-best(또는 final-iter)만. test-best 인용 금지 — 발견 시 빨간 callout에 격리하고 '무효' 표기."),
        L("헤드라인·게이트 판정 = 시드 매칭 페어 3개 이상. 기대 효과 <1.0이면 5페어. '기준 −0.3 이내 통과' 형식 금지(동등성 밴드 ±1σ 또는 우월성 ≥ +1.0만)."),
        L("모듈 토글은 |Δ| > 0.5 이고 예측 일치도 < 0.99 여야 유효, 아니면 no-op(4연속 오독 전례)."),
        L("온라인 가중 실험은 λ 궤적 플롯을 판정 산출물에 포함. 헤드라인은 legal-val 상위 k=3 ckpt의 test 평균 보고 검토."),
        L("증강·전처리 버그가 세대 비교를 오염시킴(ISSUE-025 radar 디코딩, ISSUE-026 ColorAugSSD) — 비교 전 학습 시점 설정 일치 확인."),
        L("트레이너 내부 val은 정본 재채점보다 +0.4~0.5 높게 나옴(B0: 65.4 vs 64.97). 판정은 정본으로만."),
        L("노션 갱신 규칙: 실험 판정이 바뀌면 같은 날 이 페이지 §4·§6과 레포 실험 계획 문서를 함께 갱신(레포 CLAUDE.md §3 규칙)."),
    ]

def sec_sources():
    return [CO([B("출처(코드·config·도구만) "), T(
        "학습 train_reliadino.py · 평가 val.py(--logit-adjust-tau) · 하네스 가드 tools/eval_harness_guard.py · MUSES 공식 채점 tools/eval_muses_official.py · "
        "특징 프로브 tools/probe_feature_info.py · P50 tools/p50_pretrain_align.py, tools/p50_gen_pseudomodal.py · 모델 semseg/models/reliadino/{p46,p52}.py · "
        "카드 config configs/*_screen40_{B0,E1,E2,E7,E7c,E12,E1M}.yaml, configs/*_P52_seed2026090{1,2}.yaml · 베이스라인 복원 third_party/dgfusion_train_restore/ · "
        "산출물 /drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/{ckpts,analysis_logs,train_logs}/ · 제출 zip /ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/ · "
        "커밋 develop 7d83c11(E0/E9/E1/E2) · f34c0a8(2일차 config) · 636e490(하네스 재동결) · 047e951(E-LoRA)"
    )], "📎", "gray_background")]

SECTIONS = [
    ("1. 문제 정의 · 목표 · 평가 프로토콜", sec_problem),
    ("2. 계보 요약과 가설 원장 (무엇을 시도했고 무엇이 남았나)", sec_lineage),
    ("3. 실험 노트 — 2026-08~09 캠페인 (문제 → 가설 → 실험 → 결과 → 판정)", sec_campaign),
    ("4. 일일 사이클 실험 카드 (2026-09-07~)", sec_cards),
    ("5. Related Works", sec_related),
    ("6. 한 것 / 할 것 (레포 실험 계획 문서 동기화)", sec_plan),
    ("7. 검출 트랙 (국가 R&D) 요약", sec_det),
    ("8. 재현성 · 판정 규약 · 출처", lambda: sec_rules() + sec_sources()),
]
OUTER = "0. 한눈에 보기"
APPENDIX = "부록 — 기존 실험 DB 뷰 · 서버 세팅 (원본 유지)"
OLD_OUTER_TITLES = ["출처", "1. 문제 정의", "2. 계보 요약", "3. 실험 노트", "4. 일일 사이클", "5. Related", "6. 한 것", "7. 검출 트랙", "8. 재현성"]  # 바깥 페이지에 남은 구 상세 h1(접두어 매칭) — 발견 시 제거

def existing_headings(pid):
    out = {}
    for b in page_blocks(pid):
        if b["type"] in ("heading_1", "heading_2", "heading_3"):
            out.setdefault(text_of(b), b["id"])
    return out

def child_pages(pid):
    return {b["child_page"]["title"]: b["id"] for b in page_blocks(pid) if b["type"] == "child_page"}

def chunked_append(pid, blocks, after_id=None):
    last = after_id
    for i in range(0, len(blocks), 100):
        body = {"children": blocks[i:i+100]}
        if last: body["after"] = last
        r = call("PATCH", f"/blocks/{pid}/children", body)
        if not ok(r):
            print("append failed", r); sys.exit(1)
        last = r["results"][-1]["id"]
        time.sleep(0.4)
    return last

def ensure_child(title, after_id=None):
    """바깥 페이지의 하위 페이지(제목 일치)를 찾거나 만든다. 반환 = 페이지 id."""
    kids = child_pages(PID)
    if title in kids:
        return kids[title]
    r = call("POST", "/pages", {"parent": {"page_id": PID},
                                "properties": {"title": {"title": [rt(title)]}}, "children": []})
    if not ok(r):
        print("create child failed", r); sys.exit(1)
    return r["id"]

def set_page_content(pid, blocks):
    """하위 페이지 본문 전체 교체(멱등)."""
    for b in page_blocks(pid):
        call("DELETE", f"/blocks/{b['id']}")
    chunked_append(pid, blocks)

def remove_outer_section(title):
    hid, body = section_range(PID, title, "heading_1")
    if hid is None:
        return
    for bid in body:
        call("DELETE", f"/blocks/{bid}")
    call("DELETE", f"/blocks/{hid}")

def outer_blocks(child_ids):
    b = sec_summary()
    b.append(h2("상세 페이지 (목차)"))
    for title, _ in SECTIONS:
        b.append(link_page(child_ids[title]))
    b.append(h2("핵심 그림 2장 (나머지 그림은 각 상세 페이지)"))
    b.append(IMG("fig1_sota_gap.png", "3개 벤치에서 SOTA 대비 현재 위치. 양수 = 우리가 앞섬."))
    b.append(IMG("fig7_daily_cards.png", "일일 카드 40ep 스크린 궤적(트레이너 val, 판정용 아님)과 사전 등록 판정 기준."))
    return b

def main():
    # 1) 상세 절 → 하위 페이지 (없으면 생성, 있으면 본문 교체)
    child_ids = {}
    for title, fn in SECTIONS:
        cid = ensure_child(title)
        set_page_content(cid, fn())
        child_ids[title] = cid
        print("child ok:", title)
        time.sleep(0.3)
    # 2) 바깥 페이지에 남아 있는 구 상세 절 제거
    for title, _ in SECTIONS:
        remove_outer_section(title)
    for t in OLD_OUTER_TITLES:
        remove_outer_section(t)
    # 3) 바깥 페이지 요약 절 교체(없으면 첫 블록 뒤에 생성)
    heads = existing_headings(PID)
    if OUTER in heads:
        r = replace_section(PID, OUTER, outer_blocks(child_ids), level="heading_1")
        print("outer replaced", "ok" if ok(r) else r)
    else:
        anchor = page_blocks(PID)[0]["id"]
        chunked_append(PID, [h1(OUTER)] + outer_blocks(child_ids), after_id=anchor)
        print("outer created")
    if APPENDIX not in existing_headings(PID):
        print("WARNING: appendix heading missing")
    # 4) 하위 페이지 목차 블록이 요약 절 안에 있으므로 child_page 블록 자체는 바깥 페이지 끝에 남는다 — 부록 위로 정렬은 노션이 자동 처리하지 않음(수동 이동 가능)
    hits = {"src": [], "tone": []}
    for pid in [PID] + list(child_ids.values()):
        h = audit(pid); hits["src"] += h["src"]; hits["tone"] += h["tone"]
    print("audit:", hits)

if __name__ == "__main__":
    main()
