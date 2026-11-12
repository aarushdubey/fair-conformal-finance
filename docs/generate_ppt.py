#!/usr/bin/env python3
"""Generate a 12-slide project presentation for FairTransCP — light theme."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os

# ---------------------------------------------------------------------------
# Light theme palette — warm, human feel (no generic AI blue)
# ---------------------------------------------------------------------------
BG        = RGBColor(0xFA, 0xF8, 0xF5)   # warm off-white
ACCENT    = RGBColor(0xC0, 0x39, 0x2B)   # deep terracotta
DARK      = RGBColor(0x1A, 0x1A, 0x1A)   # near-black body text
MUTE      = RGBColor(0x6B, 0x63, 0x5C)   # warm grey for subtitles
HIGHLIGHT = RGBColor(0xD3, 0x54, 0x00)   # burnt sienna for emphasis
SOFT_BG   = RGBColor(0xF0, 0xEB, 0xE5)   # slightly darker warm bg for panels
TABLE_HDR = RGBColor(0x3D, 0x32, 0x2F)   # dark brown-grey for table headers
BAR_GREY  = RGBColor(0x7F, 0x8C, 0x8D)   # neutral bar
BAR_RED   = RGBColor(0xE7, 0x4C, 0x3C)   # paradox highlight
BAR_BLUE  = RGBColor(0x34, 0x98, 0xDB)   # Generic FC


prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)


def blank():
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = BG
    return sl


def txt(sl, l, t, w, h, body, sz=22, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = body; p.font.size = Pt(sz); p.font.bold = bold
    p.font.color.rgb = color; p.alignment = align
    return tf


def bullets(sl, l, t, w, h, items, sz=18, color=DARK, spacing=1.4):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "▸  " + item
        p.font.size = Pt(sz); p.font.color.rgb = color
        p.space_after = Pt(8); p.line_spacing = spacing
    return tf


# ========================== Slide 1: Title =================================
sl = blank()
txt(sl, 0.5, 1.2, 12.3, 1.2, "FairTransCP", 48, bold=True, color=ACCENT)
txt(sl, 0.5, 2.4, 12.3, 0.8,
    "Fair Conformal Classification for Financial Transactions", 26, color=DARK)
txt(sl, 0.5, 3.4, 12.3, 0.6,
    "Balancing Coverage and Set-Size Equity Across Demographic Groups", 18, color=MUTE)
txt(sl, 0.5, 5.8, 12.3, 0.5,
    "Aarush Dubey · Mahi Nandani · Anitha Rani Inturi", 16, color=MUTE)
txt(sl, 0.5, 6.3, 12.3, 0.4,
    "SRM University-AP  |  Submitted to AISTATS 2027", 14, color=MUTE)

# accent bar at bottom
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 2: Problem ==============================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "The Problem", 36, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 12.3, 5.8, [
    "Financial ML models make high-stakes decisions: credit scoring, fraud detection, loan approvals.",
    "Conformal Prediction (CP) gives statistical uncertainty guarantees — but standard CP ignores fairness.",
    "Naive fairness fixes can backfire: equalizing coverage rates may force wider prediction sets for protected groups.",
    "Applicants sent to manual review face delays, drop-offs, and exposure to subjective human bias.",
    "We call this the *coverage–equity paradox*: enforcing demographic parity shifts penalties downstream."
], sz=20, spacing=1.5)
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 3: What is CP? =========================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "What is Conformal Prediction?", 36, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 12.3, 5.8, [
    "A framework for uncertainty quantification with finite-sample, distribution-free guarantees.",
    "For a target error rate α (e.g., 0.10), it produces a set C(x) of plausible labels — not just one answer.",
    "Guarantee: P(Y ∈ C(X)) ≥ 1 − α holds over the whole population.",
    "In lending: instead of a yes/no, an applicant gets {approve}, {reject}, or {approve, reject} by confidence level.",
    "Standard CP is marginal (overall); group-conditional CP tries equal coverage per demographic — but causes our problem."
], sz=20, spacing=1.5)
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 4: The Paradox =========================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "The Coverage–Equity Paradox", 36, bold=True, color=ACCENT)
txt(sl, 0.5, 1.1, 12.3, 0.5, "Why equal coverage ≠ fair outcomes", 20, color=MUTE)

bullets(sl, 0.5, 1.8, 6.0, 5.0, [
    "Protected groups often have noisier data → higher variance in predictions.",
    "Equalizing coverage forces the system to raise the confidence bar for those groups.",
    "This inflates their prediction sets, sending more people to ambiguous {approve, reject} cases.",
    "Manual review introduces delays, lowers conversion, and opens doors to human bias.",
    "Fixing one metric makes another dimension of fairness worse."
], sz=18, spacing=1.4)

# Right panel – metric definitions
txBox = sl.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(6.0), Inches(5.0))
tf = txBox.text_frame; tf.word_wrap = True
for i, (label, val) in enumerate([
    ("Coverage Gap  Δ_cov", "|Cov₁ − Cov₀|  — must stay small"),
    ("Set-Size Disparity  D_size", "max(avg sizes) / min(avg sizes)  — 1.0 = perfect"),
    ("Average Set Size", "Labels per prediction  — lower = more actionable"),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = label + "\n   " + val
    p.font.size = Pt(17); p.font.color.rgb = DARK; p.space_after = Pt(16)
    p.font.bold = (i == 0)

rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 5: Approach ============================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Our Approach: FairTransCP", 36, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 12.3, 5.8, [
    "Jointly calibrate two objectives: coverage equity AND set-size parity.",
    "Blend group-specific and pooled calibration thresholds using interpolation parameter λ.",
    "λ = 0 → pure group-conditional CP (exact coverage parity, risk of set inflation).",
    "λ = 1 → standard marginal CP (minimizes artificial set inflation, allows coverage gaps).",
    "Optimal λ chosen via grid search on validation data balancing both objectives.",
    "Built on explainable tree ensembles (Random Forest, XGBoost, LightGBM) — regulatory compliant."
], sz=20, spacing=1.5)
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 6: Core Mechanism ======================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Core Mechanism", 36, bold=True, color=ACCENT)

# Formula
txBox = sl.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(1.5))
tf = txBox.text_frame; p = tf.paragraphs[0]
p.text = "q̃^(g)(λ) = (1 − λ) · q̂^(g)  +  λ · q̂_pooled"
p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = HIGHLIGHT
p.alignment = PP_ALIGN.CENTER

sub = sl.shapes.add_textbox(Inches(1.5), Inches(3.1), Inches(10.3), Inches(0.5))
t = sub.text_frame; p2 = t.paragraphs[0]
p2.text = "q̂^(g)  = group-specific quantile     q̂_pooled  = overall quantile     λ ∈ [0, 1]"
p2.font.size = Pt(16); p2.font.color.rgb = MUTE; p2.alignment = PP_ALIGN.CENTER

bullets(sl, 1.5, 3.9, 10.3, 3.0, [
    "λ ≈ 0.3–0.5 in practice dampens threshold spikes on high-variance groups.",
    "Theoretical guarantee: worst-group coverage loss is bounded by λ · δ_a (linear bound).",
    "Satisfies both distribution-free validity and operational fairness simultaneously."
], sz=18, spacing=1.4)
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 7: Baselines Table =====================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Baselines We Compare Against", 36, bold=True, color=ACCENT)

tbl = [
    ("Method",              "Coverage",            "Set-Size Disparity", "Notes"),
    ("Standard CP",         "Marginal only",       "~1.03",              "No fairness enforcement"),
    ("Mondrian CP",         "Per-group exact",     "Up to 1.64",         "Pure group-conditional — triggers paradox"),
    ("LC-CP (Label Clustered)", "Group-ish",         "Moderate",           "Clusters labels before calibration"),
    ("Generic Fair CP",     "Perfect parity",      "1.00",               "Reports {0,1} for nearly every case — uninformative"),
    ("FairTransCP (Ours)",  "Near-parity",         "~1.01–1.04",         "Best balance: informative + fair"),
]
left, top, rh, cw = 0.5, 1.2, 0.65, [2.6, 2.4, 2.6, 4.3]
y = top
for ri, row in enumerate(tbl):
    x = left
    for ci, cell in enumerate(row):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(cw[ci]), Inches(rh))
        tf = s.text_frame; p = tf.paragraphs[0]
        p.text = cell
        is_hdr = ri == 0
        p.font.size = Pt(13 if is_hdr else 12)
        p.font.bold = is_hdr
        p.font.color.rgb = TABLE_HDR if is_hdr else (ACCENT if "FairTransCP" in cell else DARK)
        x += cw[ci]
    y += rh
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 8: German Credit Results ===============
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Results: German Credit (Random Forest)", 32, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 6.0, 5.5, [
    "Target coverage: 90% (α = 0.10). N = 1,000 transactions.",
    "Mondrian CP closed the coverage gap, but disparity jumped from 1.03 → 1.06 (+6.5%).",
    "Generic Fair CP achieved perfect parity at 1.00 — but average set size = 2.0. Every prediction ambiguous. Not usable.",
    "FairTransCP held disparity at 1.03, worst-group coverage at 96.1%, avg set size 1.91 — informative and fair."
], sz=18, spacing=1.4)

positions = [
    ("Standard CP",   1.032, BAR_GREY),
    ("Mondrian CP",   1.059, BAR_RED),
    ("Generic FC",    1.003, BAR_BLUE),
    ("FairTransCP",   1.026, ACCENT),
]
bl, bt, bh = 1.5, 0.45, 0.40
for name, disp, col in positions:
    w = Inches((disp - 1.0) / 0.06 * 4.5)
    rect = sl.shapes.add_shape(1, Inches(bl), Inches(bt), w, Inches(bh))
    rect.fill.solid(); rect.fill.fore_color.rgb = col; rect.line.fill.background()
    txt(sl, bl + 0.1, bt - 0.18, 2.5, 0.3, name, 13, color=DARK)
    txt(sl, bl + w + 0.1, bt + 0.02, 1.5, 0.3, f"{disp:.3f}", 13, bold=True, color=col)
    bt += 0.60

txt(sl, 7.0, 4.2, 4.5, 0.35, "Set-Size Disparity Ratio  (lower is better)", 13, color=MUTE)


# ========================== Slide 9: Adult Income Results ================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7,
    "Results: Adult Income (XGBoost) — The 59.8% Disparity Spike", 30, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 6.0, 5.5, [
    "N = 48,842 records; sex as sensitive attribute.",
    "Mondrian CP caused a 59.8% surge in set-size disparity: ratio jumped from 1.04 to 1.64!",
    "Protected-group applicants received nearly twice as many ambiguous {approve, reject} predictions.",
    "FairTransCP brought disparity back down to 1.04 — matching standard CP — while keeping worst-group coverage at 95.3%.",
    "LC-CP reduced disparity but dropped worst-group coverage to 89.4%, below the 90% safety floor."
], sz=18, spacing=1.4)

tbl2 = [
    ("Method",       "Disparity",  "Worst Cov.",  "Avg Set Size"),
    ("Standard CP",  "1.040",      "95.3%",       "1.66"),
    ("Mondrian CP",  "1.637 ⚠",    "90.9%",       "1.47"),
    ("LC-CP",        "1.034",      "89.4% ⚠",     "1.68"),
    ("Generic FC",   "1.000",      "100%",        "2.00 ✗"),
    ("FairTransCP",  "1.040 ✓",    "95.3% ✓",     "1.66 ✓"),
]
txBox = sl.shapes.add_textbox(Inches(6.8), Inches(1.5), Inches(6.0), Inches(5.0))
tf = txBox.text_frame; tf.word_wrap = True
for ri, row in enumerate(tbl2):
    for ci, cell in enumerate(row):
        p = tf.paragraphs[0] if (ri == 0 and ci == 0) else tf.add_paragraph()
        p.text = cell
        p.font.size = Pt(13)
        p.font.bold = (ri == 0)
        if "FairTransCP" in cell:
            p.font.color.rgb = ACCENT
        elif "⚠" in cell or "✗" in cell:
            p.font.color.rgb = BAR_RED
        else:
            p.font.color.rgb = DARK if ri > 0 else TABLE_HDR
        p.space_after = Pt(5)


# ========================== Slide 10: Pareto Insight =====================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Pareto Frontier: Coverage vs. Disparity", 36, bold=True, color=ACCENT)
txt(sl, 0.5, 1.1, 12.3, 0.5,
    "FairTransCP consistently lands in the sweet spot across all three benchmarks.", 18, color=MUTE)

bullets(sl, 0.5, 1.8, 6.0, 5.2, [
    "Every benchmark (German Credit, Taiwan Default, Adult Income) shows the same pattern.",
    "Generic Fair CP reaches perfect parity but sacrifices all information — useless in practice.",
    "LC-CP keeps sets tight but sometimes lets coverage slip below the 90% safety floor.",
    "Mondrian CP enforces coverage equality but explodes set-size disparity — exactly the paradox.",
    "FairTransCP finds the balance: near-perfect parity with informative, actionable predictions."
], sz=18, spacing=1.4)

txBox = sl.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(6.0), Inches(5.0))
tf = txBox.text_frame; tf.word_wrap = True
notes = [
    ("Y-axis", "worst-group coverage  (higher = better)"),
    ("X-axis", "set-size disparity ratio  (lower = better)"),
    ("", ""),
    ("Ideal zone", "top-left: high coverage, low disparity"),
    ("", ""),
    ("Mondrian CP", "far right — high disparity ⚠"),
    ("Generic FC",  "bottom — zero predictive value ✗"),
    ("FairTransCP", "closest to the ideal ✓"),
]
for i, (label, val) in enumerate(notes):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = f"{label}: {val}" if label else ""
    p.font.size = Pt(16)
    if "FairTransCP" in label:
        p.font.color.rgb = ACCENT; p.font.bold = True
    elif "⚠" in val or "✗" in val:
        p.font.color.rgb = BAR_RED
    else:
        p.font.color.rgb = DARK
    p.space_after = Pt(10)


# ========================== Slide 11: Why It Matters =====================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Why This Matters in Practice", 36, bold=True, color=ACCENT)

impacts = [
    ("Regulatory compliance",
     "ECOA & FCRA require clear adverse-action notices. Vague {approve, reject} sets violate the spirit of these laws."),
    ("Applicant experience",
     "Manual review queues cause weeks of delay; borrowers routinely drop applications and seek alternatives elsewhere."),
    ("Human-in-the-loop bias",
     "Ambiguous cases land with loan officers who exercise discretionary judgment — studies show this can favor non-protected applicants."),
    ("Algorithmic accountability",
     "Interpretable tree ensembles (RF, XGBoost) ensure decisions can be audited, explained, and challenged."),
]
y = 1.2
for title, desc in impacts:
    txt(sl, 0.5, Inches(y), 3.0, 0.4, title, 18, bold=True, color=ACCENT)
    txt(sl, 3.5, Inches(y), 9.0, 0.8, desc, 16, color=DARK)
    y += 1.2
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


# ========================== Slide 12: Summary ============================
sl = blank()
txt(sl, 0.5, 0.3, 12.3, 0.7, "Summary", 36, bold=True, color=ACCENT)
bullets(sl, 0.5, 1.2, 12.3, 5.8, [
    "Identified the coverage–equity paradox: forcing equal coverage inflates ambiguous predictions for protected groups.",
    "Proposed FairTransCP — a single interpolation parameter λ that jointly balances coverage equity and set-size parity.",
    "Proved a theoretical lower bound: worst-group coverage loss is linearly bounded by λ.",
    "Demonstrated up to 59.8% reduction in set-size disparity spikes while maintaining >90% coverage across 3 benchmarks.",
    "Built on interpretable tree ensembles — ready for real financial-regulation audits.",
    "Code: github.com/aarushdubey/fair-conformal-finance  ·  Full paper submitted to AISTATS 2027"
], sz=20, spacing=1.5)
rect = sl.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT; rect.line.fill.background()


out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FairTransCP_Presentation.pptx")
prs.save(out)
print(f"Saved → {out}")
