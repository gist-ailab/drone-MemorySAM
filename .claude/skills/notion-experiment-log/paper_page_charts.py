"""노션 논문 페이지용 정량 차트 (2026-09-08). 수치 출처 = 레포 정본 문서(레지스트리·판정 문서)의 legal 값만."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get("PAPER_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_paper_figs"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a95", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
# 팔레트(dataviz 기본, 검증됨): 1 blue 2 orange 3 aqua 4 yellow 5 magenta 6 green 7 violet 8 red
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTE = "#0b0b0b", "#52514e", "#8a8985"

def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("saved", p)

# 한글 폰트 (있으면)
import matplotlib.font_manager as fm
_fp = "/usr/share/fonts/truetype/baekmuk/dotum.ttf"
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ───────────────────────────── 1. 3벤치 SOTA 격차 (헤드라인)
def fig_gap():
    rows = [
        ("DELIVER test\n(우리 5-seed 평균 54.39 vs MM SAM-adapter 57.35)", -2.96),
        ("DELIVER test\n(우리 최고 단일런 55.29 vs 57.35)", -2.06),
        ("MUSES test\n(우리 79.79 vs 카메라단독 1위 GtA 82.39)", -2.60),
        ("MUSES test\n(우리 79.79 vs 융합계보 1위 DGFusion 79.5)", +0.29),
        ("MCubeS test\n(우리 3-seed 평균 58.07 vs Mul-VMamba 54.65)", +3.42),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    cols = [C[2] if v >= 0 else C[1] for v in vals]
    ax.barh(y, vals, color=cols, height=0.55)
    ax.axvline(0, color=INK2, lw=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=11, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlabel("mIoU 격차 (우리 - 비교 대상), 양수 = 우리가 앞섬")
    ax.set_xlim(-4.2, 4.6)
    ax.set_title("3개 벤치마크에서 SOTA 대비 현재 위치 (2026-09-08, 합법 ckpt 기준)", color=INK, fontsize=12, loc="left")
    save(fig, "fig1_sota_gap.png")

# ───────────────────────────── 2. DELIVER 캠페인 legal test (val.py native-GT)
def fig_deliver_campaign():
    items = [
        ("P46 C3-only 기준 레시피\n5-seed 평균 (N6 재선택)", 54.39, 0.76, C[0]),
        ("P46 C3-only 최고 단일런\n(seed816)", 55.29, 0, C[0]),
        ("seed821 기준런\n(카드 스크린 대조군 원본)", 53.57, 0, C[0]),
        ("+ P50 정렬 사전학습 init\n(Phase1, 페어 1개)", 54.95, 0, C[2]),
        ("+ P50-EXT 500k 스케일업 init\n(ep30 조기판정)", 53.10, 0, C[1]),
        ("+ P51 인코딩-시간 결합\n(CMLC on)", 54.58, 0, C[1]),
        ("P51 매칭 off 팔", 55.40, 0, C[0]),
        ("믹서 = 산술평균 융합\n(N2, gated-MLP 대체)", 55.45, 0, C[3]),
        ("RGB-D 2모달만\n(SOTA 최고 구성 재현)", 54.14, 0, C[1]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, e, c) in zip(y, items):
        ax.barh(yi, v, color=c, height=0.58, xerr=e if e else None,
                error_kw=dict(ecolor=INK2, capsize=3, lw=1))
        ax.text(v + 0.08, yi, f"{v:.2f}" + (f" ±{e:.2f}" if e else ""), va="center", color=INK, fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9, color=INK)
    ax.axvline(57.35, color=C[7], lw=1.4, ls="--"); ax.text(57.38, y[0] + 0.55, "MM SAM-adapter 57.35 (native GT, 동일 자)", color=C[7], fontsize=9, va="bottom")
    ax.axvline(56.71, color=MUTE, lw=1.2, ls=":"); ax.text(56.68, y[0] + 0.55, "DGFusion 56.71 (리사이즈 GT, 낙관 자)", color=MUTE, fontsize=9, ha="right", va="bottom")
    ax.set_xlim(52.5, 58.2); ax.set_xlabel("DELIVER test mIoU (val.py native-GT, val-best ckpt = 합법)")
    ax.set_title("DELIVER 캠페인(2026-08~09): 무엇이 올렸고 무엇이 내렸나", color=INK, fontsize=12, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[0], label="기준선·대조군"), Patch(color=C[2], label="이득(게이트 통과)"),
                       Patch(color=C[1], label="손해(반증)"), Patch(color=C[3], label="동급(우위 주장 철회)")],
              loc="lower right", fontsize=9, frameon=False)
    save(fig, "fig2_deliver_campaign.png")

# ───────────────────────────── 3. MUSES 공식 test 리더보드
def fig_muses():
    items = [
        ("GtA (camera-only, Codabench 1위)", 82.39, MUTE),
        ("MM SAM-adapter (RGB-L)", 81.07, MUTE),
        ("우리 P39.1-rank seed2 3모달 (img/lidar/event)", 79.788, C[0]),
        ("우리 P39.1 4모달 (+radar)", 79.571, C[0]),
        ("우리 P39.1 2모달 (RGB-L)", 79.571, C[0]),
        ("DGFusion (4모달)", 79.5, MUTE),
        ("우리 P43 PanopticDual", 79.351, C[0]),
        ("우리 P38-m2f", 79.025, C[0]),
        ("우리 P46-C3 이식 (MUSES)", 79.023, C[1]),
        ("우리 P34 3모달", 78.979, C[0]),
        ("우리 P39-DPC", 78.881, C[0]),
        ("우리 P47-D1 LiDAR 밀도화", 78.790, C[1]),
        ("CAFuser-CAA (4모달)", 78.5, MUTE),
        ("우리 P44-BMR", 78.429, C[1]),
        ("우리 P34 4모달 (+radar)", 78.256, C[1]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, c) in zip(y, items):
        ax.barh(yi, v, color=c, height=0.6)
        ax.text(v + 0.05, yi, f"{v:.3f}" if v < 82 and "우리" in lab else f"{v:.2f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9, color=INK)
    ax.set_xlim(77.5, 83.2); ax.set_xlabel("MUSES 공식 test mIoU (Codabench comp 14005, 단일 제출)")
    ax.set_title("MUSES 공식 test: 융합 계보 1위이나 카메라단독 1위에 -2.60", color=INK, fontsize=12, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[0], label="우리 (계보)"), Patch(color=C[1], label="우리 (반증된 시도)"), Patch(color=MUTE, label="선행연구")],
              loc="lower right", fontsize=9, frameon=False)
    save(fig, "fig3_muses_leaderboard.png")

# ───────────────────────────── 4. E0 특징 정보 프로브
def fig_e0():
    sets = ["원 백본 특징\n(LoRA off, 4탭)", "어댑터 후\n(LoRA on, 4탭)", "어댑터 후\n(마지막 블록)", "융합 트렁크 출력"]
    val = [36.86, 39.47, 44.14, 50.23]; test = [27.81, 32.24, 35.59, 45.14]
    cls = ["RailTrack", "Wall", "Water"]
    fused = [0.0, 33.3, 16.6]; depth_tap = [23.9, 39.6, 47.6]; raw_depth = [6.2, 40.0, 31.0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(4); w = 0.36
    a.bar(x - w/2, val, w, color=C[0], label="val"); a.bar(x + w/2, test, w, color=C[1], label="test")
    for xi, v, t in zip(x, val, test):
        a.text(xi - w/2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, color=INK); a.text(xi + w/2, t + 0.6, f"{t:.1f}", ha="center", fontsize=9, color=INK)
    a.set_xticks(x); a.set_xticklabels(sets, fontsize=9); a.set_ylabel("선형 프로브 mIoU (%)"); a.set_ylim(0, 58)
    a.set_title("(a) 정보량은 raw < adapted < fused 단조 증가 -> '어댑터가 버린다' 기각", fontsize=10, loc="left", color=INK)
    a.legend(frameon=False, fontsize=9)
    x2 = np.arange(3); w2 = 0.27
    b.bar(x2 - w2, fused, w2, color=C[0], label="융합 출력"); b.bar(x2, depth_tap, w2, color=C[2], label="depth 중간층 탭(어댑터 후)"); b.bar(x2 + w2, raw_depth, w2, color=MUTE, label="depth 중간층 탭(원 특징)")
    for xi, f, d, r in zip(x2, fused, depth_tap, raw_depth):
        for off, v in ((-w2, f), (0, d), (w2, r)):
            b.text(xi + off, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    b.set_xticks(x2); b.set_xticklabels(cls); b.set_ylabel("test recall (%)"); b.set_ylim(0, 56)
    b.set_title("(b) 붕괴 클래스 근거는 depth 중간층에 남아 있고 융합이 잃는다 -> E1·E3 유지", fontsize=10, loc="left", color=INK)
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("일일 카드 E0 - 특징 정보 프로브 (DELIVER, P46 seed821 val-best ckpt, 선형 헤드만 학습)", x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, "fig4_e0_probe.png")

# ───────────────────────────── 5. 검정력: 왜 3페어 규약인가
def fig_power():
    delta = np.linspace(0.3, 2.0, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for s, c, lab in ((0.5, C[2], "σ=0.5 (MCubeS 3-seed)"), (0.8, C[0], "σ=0.8 (DELIVER 5-seed 0.76)"), (1.0, C[1], "σ=1.0 (MUSES 5-seed 범위 0.92)")):
        n = 12.4 * s**2 / delta**2
        ax.plot(delta, n, color=c, lw=2, label=lab)
    npair = 6.2 * 0.6**2 / delta**2
    ax.plot(delta, npair, color=C[6], lw=2, ls="--", label="시드 매칭 페어 설계 (σ_pair≈0.6)")
    ax.axhline(3, color=MUTE, lw=1, ls=":"); ax.text(1.62, 3.3, "실제 확보 가능한 시드 수 ≈ 3", color=MUTE, fontsize=9)
    ax.axvline(0.3, color=C[7], lw=1, ls=":"); ax.text(0.32, 60, "구 P52 게이트 마진 0.3\n-> 팔당 34~138 시드 필요", color=C[7], fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1, 200); ax.set_xlim(0.3, 2.0)
    ax.set_xlabel("검출하려는 mIoU 차이 δ"); ax.set_ylabel("필요한 팔당 시드 수 n (log)")
    ax.set_title("측정 규약: 시드 2~3개로 판별 가능한 차이는 ±1.0 이상뿐 (α=0.05, 검정력 0.8)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig5_power.png")

# ───────────────────────────── 6. 가설 원장 판정 맵
def fig_ledger():
    H = [
        ("H1 학습 게이트 모달 가중", "✗"), ("H2 신뢰도->attn logit bias(RBMA)", "✗"), ("H3 추론 시 재가중", "✗"),
        ("H4 조건×클래스 전문화(오라클)", "✗"), ("H5 조건×모달 상호작용 실재", "✓"), ("H6 백본 표현력 지배(+11.6)", "✓"),
        ("H7 학습 해상도 768->1024", "✓"), ("H8 클래스 prototype 손실(C3)", "✓ 축특이"), ("H9 쿼리 경로 이중화", "✗"),
        ("H10 인스턴스 감독->PQ", "동결"), ("H11 radar 기여(MUSES)", "✗"), ("H12 백본 L->7B 확대", "✗"),
        ("H13 zero-init 비대칭 주입", "✗"), ("H14 MM-SA 주입 우위 전이", "✗"), ("H15 masked consistency(C2)", "✗"),
        ("H16 패치별 센서 선택 라우팅", "✗"), ("H17 cross-attn 트렁크 > MLP", "✗"), ("H18 단일런 대표 가능", "✗"),
        ("H19 인코딩-시간 결합(P51)", "✗"), ("H20 dose-response 예측력", "✓"), ("H21 gated-MLP > 평균 융합", "✗"),
        ("H22 어댑터 정렬 사전학습(P50)", "✓ 재현대기"), ("H23 백본 적응 깊이", "예약"), ("H24 C3 train 신호 발화", "예약"),
        ("H25 P50 스케일 역전 원인", "예약"), ("H26 라우팅 단서 존재", "예약(E0 부분 음성)"),
    ]
    col = {"✗": C[1], "✓": C[2]}
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off")
    ncol = 3
    for i, (name, v) in enumerate(H):
        r, c = divmod(i, ncol)
        x, y = c * 3.7, -r * 0.62
        k = "✓" if v.startswith("✓") else ("✗" if v.startswith("✗") else "?")
        fc = col.get(k, "#e6e6e2")
        ax.add_patch(plt.Rectangle((x, y - 0.25), 3.5, 0.5, color=fc, alpha=0.85 if k != "?" else 1.0, ec="none"))
        ax.text(x + 0.1, y, f"{name}", va="center", fontsize=9, color="white" if k != "?" else INK, fontweight="bold" if k != "?" else "normal")
        ax.text(x + 3.42, y, v, va="center", ha="right", fontsize=9, color="white" if k != "?" else INK2)
    ax.set_xlim(-0.1, 11.2); ax.set_ylim(-0.62 * 9 + 0.1, 0.4)
    nx = sum(v.startswith("✗") for _, v in H); nc = sum(v.startswith("✓") for _, v in H)
    ax.set_title(f"가설 원장 H1~H26 판정 맵 - 반증 {nx}건 · 확인 {nc}건 · 미결/예약 {len(H)-nx-nc}건 (2026-09-08)", color=INK, fontsize=12, loc="left")
    save(fig, "fig6_hypothesis_map.png")

# ───────────────────────────── 7. 일일 카드 40ep 스크린 궤적 (트레이너 val, 판정용 아님)
def fig_cards():
    ep = [5, 10, 15, 20, 25, 30, 35, 40]
    series = [
        ("B0 기준선 (변경 0)", [58.71, 63.17, 62.18, 63.91, 64.03, 65.13, 64.67, 65.40], INK2, "-"),
        ("E1 중간층 4탭 읽기", [60.98, 63.56, 66.30, 66.57, 66.43, 66.88, 67.25, None], C[0], "-"),
        ("E2 전 선형층 LoRA r32 (hpca100, legal test 54.50)", [62.54, 63.44, 60.09, 62.40, 65.25, 66.91, 67.85, 68.65], C[2], "-"),
        ("E3 센서별 클래스 prototype", [59.35, 63.72, 62.23, 65.44, 65.97, 65.51, None, None], C[3], "-"),
        ("E4 혼동 쌍 margin (auto 쌍)", [59.77, 63.17, 65.92, 65.70, 65.79, None, None, None], C[4], "-"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for lab, ys, c, ls in series:
        xs = [e for e, v in zip(ep, ys) if v is not None]; vs = [v for v in ys if v is not None]
        a.plot(xs, vs, marker="o", ms=5, lw=2, color=c, ls=ls, label=lab)
        a.text(xs[-1] + 0.4, vs[-1], f"{vs[-1]:.2f}", color=c, fontsize=9, va="center")
    a.set_xlabel("epoch (40ep 스크린, EVAL_INTERVAL 5)"); a.set_ylabel("트레이너 내부 val mIoU (%)")
    a.set_xlim(4, 43); a.set_ylim(57.5, 69)
    a.set_title("(a) DELIVER 카드 - 트레이너 val 궤적 (판정은 완주 후 legal test 하나로만)", fontsize=10, loc="left", color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    # (b) B0 정본 재채점과 통과선
    b.axis("off")
    rows = [["구분", "값"], ["B0 legal test (val.py native-GT)", "53.78"], ["B0 legal val", "64.97"],
            ["B0 트레이너 val@ep40 (참고)", "65.4 (+0.43)"], ["카드 통과선 (Δtest ≥ +1.0)", "test ≥ 54.78"],
            ["카드 폐기선", "test < 54.28 또는 악조건 -0.5"], ["E2 legal (val-best ep40)", "test 54.50 / val 68.38 (Δ +0.72 회색지대)"], ["E7 / E7c MUSES PhysAug off/on", "val 80.29 / 80.18 @40 (공식 재채점 중)"],
            ["E1M MUSES 4탭 (E7 짝)", "ep10 75.55 vs E7 75.39"]]
    t = b.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left", colWidths=[0.6, 0.4])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.55)
    for (r, c_), cell in t.get_celld().items():
        cell.set_edgecolor("#e6e6e2"); cell.set_text_props(color=INK)
        if r == 0: cell.set_facecolor("#eef2f7"); cell.set_text_props(fontweight="bold")
    b.set_title("(b) 판정 기준 (사전 등록)", fontsize=10, loc="left", color=INK)
    fig.suptitle("일일 카드 프로그램 1~2일차 (2026-09-07~08) - 카드 하나 = 변수 하나 = 하루", x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, "fig7_daily_cards.png")

if __name__ == "__main__":
    fig_gap(); fig_deliver_campaign(); fig_muses(); fig_e0(); fig_power(); fig_ledger(); fig_cards()
