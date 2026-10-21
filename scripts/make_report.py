"""Build the short progress report PDF (reports/NL-Ear_progress_report.pdf).

All numbers come from the committed results notes: notes/phase2_baseline.md, notes/u1_baseline.md,
notes/phase3_killtest.md, notes/u2_killtest.md, notes/c2_baselines.md, notes/c3_cms.md,
notes/c3b_confirm.md.

    python scripts/make_report.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "reports" / "NL-Ear_progress_report.pdf"
INK, MUTED, ACCENT, RULE, HEAD_BG, GOOD, BAD = (colors.HexColor(c) for c in
    ("#1f2328", "#5b6470", "#2b5c8a", "#d0d7de", "#eef3f8", "#1a7f37", "#b42318"))

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                       textColor=INK, alignment=TA_LEFT, spaceAfter=2)
SUB = ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10)
H = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                   textColor=ACCENT, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=4)
BUL = ParagraphStyle("bl", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2.5, bulletFontName="Helvetica")
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.8, leading=11.2, spaceAfter=0)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("n", parent=BODY, fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=3)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, bold_first_col=False, highlight_rows=()):
    data = [[p(c, CELLB) for c in rows[0]]]
    for r in rows[1:]:
        data.append([p(c, CELLB if (bold_first_col and i == 0) else CELL) for i, c in enumerate(r)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f8fa")))
    t.setStyle(TableStyle(style))
    return t


def verdict(text, ok):
    color = "#1a7f37" if ok is True else ("#b42318" if ok is False else "#9a6700")
    return f'<font color="{color}"><b>{text}</b></font>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "NL-Ear progress report  |  status as of 21 October 2026")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def build():
    s = []
    s.append(p("NL-Ear: Progress Report", TITLE))
    s.append(p("Nested Learning for open-set ear recognition &nbsp;|&nbsp; Harsh Chattar, MTP &nbsp;|&nbsp; "
               "status as of 21 October 2026", SUB))

    s.append(p("1. Goal", H))
    s.append(p("Build an ear recognition system that (1) recognizes people it never saw during training, from a few "
               "enrollment photos, and (2) <b>rejects unknown people</b> who were never enrolled; and test whether "
               "<b>Nested Learning (NL)</b> (Behrouz et al., NeurIPS 2025) makes such a system better than standard "
               "methods. Every comparison uses identity-disjoint splits, several random seeds, and decision rules "
               "fixed before the results were seen."))

    s.append(p("2. What we did", H))
    s.append(table([
        ["Stage", "What", "Outcome"],
        ["Setup", "Data pipeline for AMI, EarVN1.0, AWE and UERC 2023; duplicate removal; open-set splits "
                  "(known + unknown test people)", "Done"],
        ["Baselines", "ResNet-50 + ArcFace; rejection by cosine similarity, subspace and EVM", "Done on 4 datasets"],
        ["NL idea 1", "Titans-style memory per enrolled person; its <i>surprise</i> is used to reject strangers "
                      "(untrained and meta-learned)", verdict("No gain", False) + " (EarVN and UERC)"],
        ["Continual setup", "Learn people in groups over time; measure forgetting of earlier groups",
         "Large forgetting found"],
        ["Standard fixes", "EWC, LwF, replay of a few stored photos", "Replay best, small gain"],
        ["NL idea 2", "CMS: network parts that learn at different speeds",
         verdict("CMS blocks: no gain", False) + "; slow backbone: +0.04 rank-1 on EarVN, "
         + verdict("but not on UERC", False)],
        ["NL idea 3", "M3: NL's multi-scale momentum optimizer, instead of AdamW",
         verdict("Worse than AdamW", False)],
    ], [26, 98, 50], bold_first_col=True))

    s.append(p("3. Baseline results", H))
    s.append(table([
        ["Dataset", "Training people", "Rank-1", "AUROC", "DIR@5%"],
        ["AMI (lab photos)", "70", "0.948", "0.937", "0.760"],
        ["EarVN1.0 (in the wild)", "115", "0.591", "0.716", "0.209"],
        ["AWE (in the wild, small)", "70", "0.410", "0.626", "0.105"],
        ["<b>UERC 2023</b>", "<b>917</b>", "<b>0.662</b>", "<b>0.784</b>", "<b>0.370</b>"],
        ["UERC 2023, 5 enrollment photos", "917", "0.809", "0.860", "0.536"],
    ], [58, 30, 26, 26, 26], highlight_rows=(4,)))
    s.append(p("Test sets, 2 enrollment photos per person unless noted; means over 5 seeds (3 for UERC; UERC seed-to-seed "
               "spread is at most 0.010). Datasets differ, so rows are not directly comparable. "
               "<b>Rank-1</b>: the right person is the top match. <b>AUROC</b>: how well known and unknown people are "
               "separated (1.0 perfect, 0.5 chance). <b>DIR@5%</b>: share of known people both accepted and correctly "
               "identified when only 5% of strangers are wrongly accepted.", NOTE))

    s.append(KeepTogether([p("4. Nested Learning results", H), table([
        ["Test", "Data", "Question", "Result"],
        ["Phase 3", "EarVN", "Does the NL memory's surprise reject strangers better than cosine?",
         verdict("No", False) + ": never better than cosine; meta-training never helped (58 training people)"],
        ["U2", "UERC", "Same test with 8x more meta-training data (459 people)",
         verdict("No", False) + ": 0 of 3 seeds better; so it was not a lack of data"],
        ["C3", "EarVN", "Does CMS reduce forgetting and improve accuracy in continual learning?",
         verdict("No", False) + " on the main metric; some promising side results"],
        ["C3b", "EarVN (fresh seeds)", "Do C3's promising side results replicate?",
         verdict("Mostly no", False) + ": unseen-people gain was a fluke; CMS blocks add nothing. "
         + verdict("Slow backbone: +0.04 rank-1, 5 of 5 seeds", True)],
        ["U3", "UERC", "Does the slow-backbone effect hold at large scale?",
         verdict("No", False) + ": it reverses (rank-1 -0.015 vs an identical control, 0 of 3 seeds)"],
        ["U4", "UERC", "Does NL's M3 optimizer train a better model than AdamW?",
         verdict("No", False) + ": DIR@5% 0.301 vs 0.370 (0 of 3 seeds); its NL slow momentum makes it worse"],
    ], [16, 24, 64, 70])]))

    s.append(KeepTogether([p("5. Continual learning numbers (seen-people rank-1 after all groups)", H), table([
        ["Method", "EarVN (5 groups, 5 seeds)", "UERC (10 groups, 3 seeds)"],
        ["Train on everyone at once (upper bound)", "0.809", "0.899"],
        ["Train group by group, no protection", "0.536", "0.595"],
        ["Replay (keep 5 photos per person)", "0.570", "0.720"],
        ["Replay + same-architecture control", "0.580", "0.728"],
        ["Replay + slow backbone (NL)", "0.618 (better)", "0.713 (worse)"],
    ], [78, 48, 48]), p("EarVN replay rows are from the fresh-seed confirmation (C3b: seeds 6-10), joint and naive from "
                        "seeds 1-5 (C2). All UERC arms in fp32. The slow-backbone gain on EarVN "
                        "(+0.038 vs control, 5 of 5 seeds) did not carry over to UERC (-0.015, 0 of 3 seeds).", NOTE)]))

    s.append(p("6. Main takeaways", H))
    s += bullets([
        "<b>More training people is the biggest lever.</b> On UERC, training on 458 vs 917 people moves DIR@5% from "
        "0.24 to 0.37; no NL method changed it by more than 0.01.",
        "<b>NL memories do not beat simple cosine matching</b> for rejecting strangers, on small or large data, "
        "untrained or meta-learned. This is a clear, well-tested negative result.",
        "<b>NL's multi-speed idea did not generalize:</b> a slowly-updating backbone helped retain enrolled people "
        "on the small EarVN set (+0.04 rank-1), but hurt slightly at UERC scale (-0.015).",
        "<b>All three NL families were tested</b> (memories, multi-speed architectures, the M3 optimizer); "
        "<b>none beats standard methods at scale</b>.",
        "<b>Replay works well at scale</b> (UERC rank-1 0.72 vs 0.60 without it), but training on everyone at once "
        "(0.90) is still far ahead.",
        "<b>Method:</b> fixed rules before results, fresh seeds for confirmation; this caught one false positive.",
    ])

    s.append(p("7. Next steps", H))
    s += bullets([
        "Thesis write-up: strong UERC baselines, and a rigorous, pre-registered study showing that Nested Learning "
        "does not improve open-set or continual ear recognition at scale.",
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="NL-Ear Progress Report",
                            author="Harsh Chattar", subject="Nested Learning for open-set ear recognition")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
