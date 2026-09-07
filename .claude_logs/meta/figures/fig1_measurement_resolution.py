"""Fig. 1 — 측정 분해능: 현재 P52 게이트는 시드 노이즈 아래에 있다.

Panels
  a (hero)  effect size δ vs required seeds per arm (one-sided α=0.05, power 0.8)
            for the three benchmark σ values; independent (solid) vs matched-pair (dashed).
  b         positive single-variable effects from the campaign vs the DELIVER seed-std band.
  c         UniBal laziness gap per modality: mean and full range over 300 epochs (P47-2 fixed run).

Source data: hypothesis-ledger / retrospective §4.2 (b), 2026-09-04 UniBal calibration §1 (c).
"""
import sys, json, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, "/home/jemo/.claude/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

# ── MANDATORY font + SVG rules ──────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})  # editable text in SVG/PDF
plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 6.5
plt.rcParams['ytick.labelsize'] = 6.5

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "green_2": "#AADCA9",
    "red_strong": "#B64342", "red_2": "#E9A6A1", "red_1": "#F6CFCB",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676", "neutral_dark": "#4D4D4D",
    "teal": "#42949E", "violet": "#9A4D8E",
}

def add_panel_label(ax, label, x_offset_pt=-4, y_offset_pt=3):
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + offset, fontsize=8,
            fontweight='bold', va='bottom', ha='right')

# ── data ────────────────────────────────────────────────────────────────────
Z_A, Z_B = 1.6449, 0.8416          # one-sided α=0.05, power 0.80
K_IND = 2 * (Z_A + Z_B) ** 2       # independent two-arm: n/arm = K·σ²/δ²
K_PAIR = (Z_A + Z_B) ** 2          # matched pairs:       n     = K·σ_pair²/δ²
assert abs(K_IND - 12.37) < 0.05 and abs(K_PAIR - 6.18) < 0.05

sigma_bench = {"MCubeS σ=0.49": 0.49, "DELIVER σ=0.76": 0.76, "MUSES σ=0.92": 0.92}
sigma_pair = 0.60                  # same-seed rerun spread (GPU nondeterminism)
delta = np.linspace(0.2, 2.0, 400)
assert (delta > 0).all() and sigma_pair > 0  # positivity guard for log-scale y

effects = [  # (label, low, high, family)  family: rep=representation/info, sig=training signal, alloc=allocation, aug=outside fairness line
    ("SAM2 → DINOv3-L backbone", 11.6, 11.6, "rep"),
    ("Backbone S+ → L", 8.82, 8.82, "rep"),
    ("Eval resolution 768 → 1024", 1.08, 2.01, "rep"),
    ("C3 prototype loss (DELIVER)", 1.35, 1.74, "sig"),
    ("PhysAug (unfair recipe)", 1.12, 1.12, "aug"),
    ("P39.1 rank repair (MUSES)", 0.76, 0.76, "alloc"),
    ("P50 pretraining (1 pair)", 0.74, 0.74, "rep"),
    ("P36 router", 0.10, 0.10, "alloc"),
]
fam_color = {"rep": PALETTE["blue_main"], "sig": PALETTE["teal"],
             "alloc": PALETTE["violet"], "aug": PALETTE["neutral_light"]}
noise_lo, noise_hi = 0.76, 2.21   # DELIVER true-seed std: after driver unification / at 08-20 verdict

gap = [  # (modality, mean, lo, hi, λ_u at CAP 0.7)
    ("RGB", -0.583, -0.607, -0.551, 0.0),
    ("LiDAR", -0.082, -0.099, -0.063, 0.0),
    ("Event", 0.082, 0.060, 0.108, 0.046),
    ("Radar", 0.583, 0.539, 0.636, 0.333),
]
CAP = 0.7

# ── layout: a spans both rows on the left; b (top) and c (bottom) on the right ──
MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 78 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1],
                      left=0.07, right=0.985, top=0.93, bottom=0.14, wspace=0.50, hspace=0.80)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# ── a: power curves ─────────────────────────────────────────────────────────
line_colors = [PALETTE["green_3"], PALETTE["blue_main"], PALETTE["red_strong"]]
for (lab, s), col in zip(sigma_bench.items(), line_colors):
    n_ind = K_IND * s ** 2 / delta ** 2
    keep = n_ind >= 1.0
    ax_a.plot(delta[keep], n_ind[keep], color=col, lw=1.4, label=lab)
n_pair = K_PAIR * sigma_pair ** 2 / delta ** 2
keep_p = n_pair >= 1.0
ax_a.plot(delta[keep_p], n_pair[keep_p], color=PALETTE["neutral_dark"], lw=1.2, ls="--",
          label=f"matched pairs, σ_pair={sigma_pair:.1f}")
ax_a.set_yscale("log")
ax_a.set_ylim(1, 400)
ax_a.set_xlim(0.2, 2.0)
ax_a.set_xlabel("Effect size to detect, δ (mIoU points)")
ax_a.set_ylabel("Seeds required per arm (one-sided α = 0.05, power 0.8)")
# planned / proposed seed counts
for n, txt, col in [(2, "P52 plan: 2 seeds", PALETTE["red_strong"]), (3, "R1 rule: 3 seeds", PALETTE["neutral_dark"])]:
    ax_a.plot([0.86, 2.0], [n, n], color=col, lw=0.7, ls=":", zorder=0)
    ax_a.text(0.33, n, txt, ha="left", va="center", fontsize=6, color=col)
# gate margin marker
ax_a.plot([0.3, 0.3], [1, 400], color=PALETTE["red_strong"], lw=0.7, ls=":", zorder=0)
ax_a.text(0.34, 380, "G1–G3 margin 0.3", fontsize=6, color=PALETTE["red_strong"], va="top", ha="left")
ax_a.axvspan(0.3, 1.0, color=PALETTE["red_1"], alpha=0.35, zorder=0, lw=0)
ax_a.legend(loc="upper right", fontsize=6, handlelength=1.8, bbox_to_anchor=(1.0, 0.99))
ax_a.set_yticks([1, 3, 10, 30, 100, 300]); ax_a.set_yticklabels(["1", "3", "10", "30", "100", "300"])
ax_a.minorticks_off()
add_panel_label(ax_a, "a")

# ── b: effect sizes vs noise band ───────────────────────────────────────────
labels = [e[0] for e in effects][::-1]
mid = np.array([(e[1] + e[2]) / 2 for e in effects])[::-1]
lo = np.array([e[1] for e in effects])[::-1]
hi = np.array([e[2] for e in effects])[::-1]
cols = [fam_color[e[3]] for e in effects][::-1]
y = np.arange(len(labels))
ax_b.barh(y, mid, color=cols, height=0.62, zorder=2)
rng = (hi - lo) > 0
ax_b.errorbar(mid[rng], y[rng], xerr=np.vstack([(mid - lo)[rng], (hi - mid)[rng]]),
              fmt="none", ecolor=PALETTE["neutral_dark"], elinewidth=0.7, capsize=1.5, zorder=3)
ax_b.axvspan(noise_lo, noise_hi, color=PALETTE["red_1"], alpha=0.45, zorder=1, lw=0)
ax_b.text(noise_hi + 0.25, -0.55, "band: DELIVER seed std 0.76–2.21", ha="left", va="top",
          fontsize=5.5, color=PALETTE["red_strong"])
for i, (yi, m, h, l) in enumerate(zip(y, mid, hi, lo)):
    xt = max(h, noise_hi) + 0.15
    ax_b.text(xt, yi, f"+{m:.2f}" if h == l else f"+{l:.2f}–{h:.2f}",
              va="center", fontsize=5.5, color=PALETTE["neutral_dark"])
ax_b.set_yticks(y); ax_b.set_yticklabels(labels, fontsize=5.6)
ax_b.set_xlim(0, 13.5)
ax_b.set_xlabel("Legal test mIoU gain from one variable (points)")
ax_b.set_ylim(-1.45, len(labels) - 0.3)
add_panel_label(ax_b, "b")

# ── c: UniBal laziness gap constancy ────────────────────────────────────────
mods = [g[0] for g in gap]
yc = np.arange(len(mods))[::-1]
for (m, mean, glo, ghi, lam), yy in zip(gap, yc):
    col = PALETTE["blue_main"] if lam > 0 else PALETTE["neutral_mid"]
    ax_c.plot([glo, ghi], [yy, yy], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
    ax_c.plot(mean, yy, "o", color=col, ms=3.2, zorder=3)
    ax_c.text(0.80, yy, f"λ_u = {lam:.2f}" if lam > 0 else "λ_u = 0", va="center", fontsize=5.8,
              color=col)
ax_c.axvline(0, color=PALETTE["neutral_light"], lw=0.7, zorder=0)
ax_c.axvline(CAP, color=PALETTE["red_strong"], lw=0.7, ls=":", zorder=0)
ax_c.text(CAP + 0.02, len(mods) - 0.55, "CAP 0.7", ha="left", va="bottom", fontsize=5.5, color=PALETTE["red_strong"])
ax_c.set_yticks(yc); ax_c.set_yticklabels(mods, fontsize=6.2)
ax_c.set_xlim(-0.75, 1.18)
ax_c.set_ylim(-0.6, len(mods) - 0.2)
ax_c.set_xlabel("Laziness gap (mean, range over 300 epochs)")
add_panel_label(ax_c, "c")

fig.canvas.draw()
out = "/home/jemo/.claude/jobs/3a914e7f/tmp/fig1_measurement_resolution"
require_matplotlib_panel_alignment(
    fig, json_out=f"{out}.alignment.json", overlay_svg=f"{out}.alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    panel_ids={ax_a: "a", ax_b: "b", ax_c: "c"},
    column_groups=[["b", "c"]],
    exemptions=[{"panels": ["a"], "checks": ["panel-width", "row", "horizontal-gutter"],
                 "reason": "panel a is a deliberate 2-row hero spanning both rows of the right column"}],
)
fig.savefig(f"{out}.svg")
fig.savefig(f"{out}.pdf")
fig.savefig(f"{out}.png", dpi=300)
fig.savefig(f"{out}.tiff", dpi=600)
print("saved", out)
