"""
Generate Cardiology-only (CAR0001-CAR0005) Evaluation Deck
==================================================
Reuses the slide builders from generate_res0050_gemini36_deck.py and adds a
metrics slide comparing the 3 models on CAR0001.
  1. Transcript comparison (real vs STT)
  2. Summary comparison
  3. SOAP table (Ground Truth vs Predicted)
  4. Model metrics (clinical fidelity / logical consistency / SOAP structure)
"""

import csv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN

import generate_res0050_gemini36_deck as d

METRICS_CSV = d.BASE_DIR / 'model_comparison_metrics.csv'


def build_metrics_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    d.set_slide_bg(slide, d.BG_DARK)
    d.add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.6),
                   f'Slide 4: Model Metrics Comparison — {d.CASE_LABEL}',
                   font_size=24, color=d.ACCENT_BLUE, bold=True)
    d.add_text_box(slide, Inches(0.5), Inches(0.85), Inches(12.3), Inches(0.35),
                   'LLM-judge scores (0–100) for each model on the cardiology case',
                   font_size=14, color=d.TEXT_GRAY)

    with open(METRICS_CSV, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['case_id'] == d.CASE_ID]

    cd = CategoryChartData()
    cd.categories = ['Clinical Fidelity', 'Logical Consistency', 'SOAP Structural Fidelity']
    for r in rows:
        cd.add_series(r['model'], [int(r['clinical_fidelity']),
                                   int(r['logical_consistency']),
                                   int(r['soap_structural_fidelity'])])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5),
                                Inches(12.1), Inches(5.6), cd)
    ch = gf.chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.color.rgb = d.TEXT_WHITE
    ch.legend.font.size = Pt(14)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 80, 100
    va.tick_labels.font.color.rgb = d.TEXT_GRAY
    va.major_gridlines.format.line.color.rgb = d.BORDER_COLOR
    ch.category_axis.tick_labels.font.color.rgb = d.TEXT_WHITE
    ch.category_axis.tick_labels.font.size = Pt(14)
    colors = [d.ACCENT_BLUE, d.ACCENT_GREEN, d.ACCENT_ORANGE]
    plot = ch.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.color.rgb = d.TEXT_WHITE
    plot.data_labels.font.size = Pt(12)
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c


CASES = ['CAR0001', 'CAR0002', 'CAR0003', 'CAR0004', 'CAR0005']
OUTPUT_FILE = d.BASE_DIR / 'Cardiology_All_Cases_Presentation.pptx'


def main():
    prs = Presentation()
    prs.slide_width = d.SLIDE_WIDTH
    prs.slide_height = d.SLIDE_HEIGHT

    for case in CASES:
        d.CASE_ID = case
        d.CASE_LABEL = f'{case}: Cardiology'
        stt_text = d.read_diarized_transcript(case)
        gt_text = d.read_clean_transcript(case)
        pred = d.read_soap_csv(d.PREDICTED_DIR / f'{case}_soap.csv')
        gt = d.read_soap_csv(d.GROUND_TRUTH_DIR / f'{case}_soap.csv')

        d.build_dialogue_slide(prs, stt_text, gt_text)
        d.build_summary_slide(prs, pred.get('summary', ''), gt.get('summary', ''))
        d.build_soap_table_slide(prs, pred, gt)

    prs.save(str(OUTPUT_FILE))
    print(f'Saved: {OUTPUT_FILE} ({len(prs.slides)} slides)')


if __name__ == '__main__':
    main()
