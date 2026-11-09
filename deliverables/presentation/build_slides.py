"""Build the 7-minute talk (stl_agent_talk.pptx) on the UCF template.

Every number comes from the report (../report/sec/5_results.tex), which takes them from
eval/results/report-v1.md. Figures are rendered from the report's PDFs with pdftoppm.

Needs python-pptx (not a pixi dependency; the app doesn't use it):
    python -m venv /tmp/venv && /tmp/venv/bin/pip install python-pptx
    /tmp/venv/bin/python deliverables/presentation/build_slides.py
"""

import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "UCF-PowerPoint-Template-Brand2022-Helvetica-9-22-22-FinalV2.potx"
FIGURES = HERE.parent / "report" / "figures"
OUT = HERE / "stl_agent_talk.pptx"

GOLD = RGBColor(0xF8, 0xC3, 0x23)
INK = RGBColor(0x1A, 0x19, 0x18)
GRAY = RGBColor(0x50, 0x50, 0x50)
LIGHT = RGBColor(0xF0, 0xF0, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"  # the template's body face; Helvetica is not on every machine


def open_template(path):
    """A .potx is a .pptx with another main content type."""
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(b"presentationml.template.main+xml", b"presentationml.presentation.main+xml")
            zout.writestr(item, data)
    buf.seek(0)
    prs = Presentation(buf)
    # Drop the template's sample slides (unreferenced parts are not saved).
    ids = prs.slides._sldIdLst
    for sld in list(ids):
        prs.part.drop_rel(sld.rId)
        ids.remove(sld)
    return prs


def layout(prs, name):
    for master in prs.slide_masters:
        for lay in master.slide_layouts:
            if lay.name == name:
                return lay
    raise KeyError(name)


def render(name, dpi, tmp):
    """Render a report figure PDF to PNG."""
    out = tmp / name
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", FIGURES / f"{name}.pdf", out], check=True)
    return out.with_suffix(".png")


def drop_placeholder(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            ph._element.getparent().remove(ph._element)
            return


def set_text(shape, text):
    """Fill a placeholder, keeping its formatting (str or list of lines)."""
    lines = [text] if isinstance(text, str) else text
    tf = shape.text_frame
    tf.text = lines[0]
    for line in lines[1:]:
        tf.add_paragraph().text = line


def text_box(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, name=None):
    """runs: a str, or a list of paragraphs; each paragraph a str or a list of (text, overrides) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(size * 0.5)
        for text, opts in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = FONT
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.color.rgb = opts.get("color", color)
    return box


def card(slide, x, y, w, h, fill=LIGHT, name=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    if name:
        shape.name = name
    return shape


def stat(slide, x, y, w, number, label, num_size=40, color=INK):
    """A big number with a short label under it."""
    text_box(slide, x, y, w, 0.75, number, size=num_size, bold=True, color=color)
    text_box(slide, x, y + 0.8, w, 0.7, label, size=14, color=GRAY)


def picture(slide, path, x, y, w=None, h=None, alt=""):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def picture_fit(slide, path, x, y, w, h, alt="", left=False):
    """Largest picture with the image's aspect ratio inside the box, centred (or left-aligned)."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return picture(slide, path, x if left else x + (w - pw) / 2, y + (h - ph) / 2, pw, ph, alt)


def content_slide(prs, title, notes):
    s = prs.slides.add_slide(layout(prs, "Content - Full Width"))
    s.shapes.title.text = title
    drop_placeholder(s, 1)
    s.notes_slide.notes_text_frame.text = notes
    return s


def build():
    if not TEMPLATE.exists():
        raise SystemExit(f"{TEMPLATE.name} is missing: download it as described in {HERE / 'README.md'}")
    prs = open_template(TEMPLATE)
    tmp = Path(tempfile.mkdtemp())
    fig = {n: render(n, 300, tmp) for n in ("pipeline_preview", "models", "attempts", "organic", "failures", "cost")}

    # 1. Title
    s = prs.slides.add_slide(layout(prs, "Title Slide - Yellow"))
    s.shapes.title.text = "Text to Printable Parts"
    set_text(s.placeholders[1], ["Which knowledge helps an LLM write, check and repair CAD code?",
                                 "Michael Laager  ·  CAP 6938 AI Knowledge Systems  ·  Fall 2026"])
    s.notes_slide.notes_text_frame.text = (
        "Hi, I'm Michael. My project is an agent that turns a plain-language request into a 3D-printable STL file. "
        "The question behind it: which kinds of knowledge actually help a language model write, check and repair CAD "
        "code, and does the answer change with the model?")

    # 2. Problem
    s = content_slide(prs, "WHY CAD CODE, NOT A MESH", (
        "3D printers are everywhere, but designing a part still means learning CAD. Most text-to-3D models output a "
        "mesh directly: it can look right, but you can't ask it for a 40 millimetre hole and trust the result. "
        "Instead, the LLM writes CadQuery, which is Python CAD code. That's exact, you can edit it, and above all you "
        "can check it: run the code, measure the solid, and send any error back to the model."))
    cols = [
        ("Text → mesh", ["Looks plausible", "No exact sizes, holes or wall thickness", "Nothing to check or fix"], LIGHT, INK),
        ("Text → CAD code (CadQuery)", ["Exact and parametric", "Editable afterwards",
                                         "Checkable: run it, measure it, repair it"], GOLD, INK),
    ]
    for i, (head, items, fill, color) in enumerate(cols):
        x = 0.79 + i * 6.03
        card(s, x, 2.35, 5.72, 3.9, fill=fill, name=f"Card {head}")
        text_box(s, x + 0.4, 2.7, 4.9, 0.6, head, size=24, bold=True, color=color)
        text_box(s, x + 0.4, 3.55, 4.9, 2.5, [[("•  " + t, {})] for t in items], size=18, color=color)
    text_box(s, 0.79, 6.4, 11.75, 0.4, "A request in plain words → a checked STL file ready for the printer",
             size=16, color=GRAY)

    # 3. Pipeline
    s = content_slide(prs, "THE AGENT", (
        "One request goes through an intake gate, a single LLM call that decides the scope and asks for missing sizes. "
        "For a mechanical part, we retrieve three similar examples from a hand-checked library of 26, the model writes "
        "a script plus the size it expects, we run it in a sandbox and check the mesh is a closed solid of the right "
        "size. Any failure goes back to the model, up to four attempts. Organic objects are sculpted with signed "
        "distance function code through the same loop, or downloaded from Thingiverse. Combined objects are built "
        "piece by piece and merged."))
    picture_fit(s, fig["pipeline_preview"], 0.79, 2.35, 11.75, 4.35,
                alt="Pipeline: request, intake gate, retrieve examples, write script, sandbox, mesh check, STL; "
                    "Thingiverse branch and combined-parts branch")

    # 4. Knowledge sources and research questions
    s = content_slide(prs, "FOUR KINDS OF KNOWLEDGE", (
        "The course is about knowledge systems, so I framed the agent as drawing on four kinds of knowledge. Retrieved "
        "examples, that's RAG. Feedback from actually running the code. Knowledge in the weights: which model, how "
        "big, and whether fine-tuning a small one helps. And an outside repository, Thingiverse, versus sculpting "
        "with code. Each one maps to a research question, plus one underneath all of them: can we trust an LLM judge "
        "to say the shape is right?"))
    kinds = [
        ("1", "Retrieved examples", "RAG over 26 hand-checked CadQuery examples (TF-IDF, top 3)",
         "Do examples help, and which models?"),
        ("2", "Execution feedback", "Script errors, mesh checks and measured size sent back",
         "How much does the repair loop add?"),
        ("3", "Model weights", "7 builders: API, local, and a 4B model fine-tuned on verified builds",
         "Is the gap size or running locally?"),
        ("4", "Outside repository", "Thingiverse download vs. sculpting with SDF code",
         "Organic shapes: sculpt or download?"),
    ]
    for i, (num, head, body, rq) in enumerate(kinds):
        x = 0.79 + i * 3.0
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(2.35), Inches(0.75), Inches(0.75))
        circ.fill.solid()
        circ.fill.fore_color.rgb = GOLD
        circ.line.fill.background()
        circ.shadow.inherit = False
        tf = circ.text_frame
        tf.text = num
        r = tf.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.name = Pt(24), True, FONT
        r.font.color.rgb = INK
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        text_box(s, x, 3.25, 2.7, 0.5, head, size=20, bold=True)
        text_box(s, x, 3.8, 2.7, 1.1, body, size=15, color=GRAY)
        card(s, x, 4.95, 2.7, 1.1)
        text_box(s, x + 0.2, 4.95, 2.3, 1.1, rq, size=15, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text_box(s, 0.79, 6.3, 11.75, 0.4, "Underneath every shape result: how far can a visual LLM judge be trusted?",
             size=15, color=GRAY)

    # 5. Evaluation setup
    s = content_slide(prs, "HOW WE MEASURED IT", (
        "The benchmark has 48 mechanical parts, each with a hand-written reference solution and expected size, and 30 "
        "organic objects. Seven builder models plus Thingiverse, three runs each: 2,118 runs in one pre-planned round. "
        "Right size is measured directly. Right shape needs a visual judge: we render six views, like this, and a "
        "multimodal model scores them. I picked the judge out of four by grading 146 parts by hand, blind. Sonnet "
        "agreed best, 96 percent held out. Our earlier, cheaper judge was at 83 and was too harsh on strong models."))
    stats = [("78", "requests: 48 mechanical\n+ 30 organic"), ("7 + 1", "builder models\n+ Thingiverse"),
             ("2,118", "runs (3 per case\nand setup)"), ("96%", "judge accuracy on\n146 hand-graded parts")]
    for i, (num, label) in enumerate(stats):
        x = 0.79 + (i % 2) * 2.85
        y = 2.4 + (i // 2) * 1.85
        stat(s, x, y, 2.6, num, label)
    picture_fit(s, FIGURES / "judge_c01_reference.png", 6.6, 2.35, 5.94, 3.7,
                alt="Six-view render of a reference part, as the visual judge sees it")
    text_box(s, 6.6, 6.15, 5.94, 0.4, "What the judge sees: six views of the finished part", size=12,
             color=GRAY, align=PP_ALIGN.CENTER)
    text_box(s, 0.79, 6.15, 5.5, 0.6, "Right size: measured.  Right shape: visual LLM judge.", size=15, bold=True)

    # 6. Which model (RQ1)
    s = content_slide(prs, "THE MODEL MATTERS MOST", (
        "First, the model. Claude Sonnet builds the right part 99 percent of the time, Haiku 94. Gemini Flash Lite "
        "and Gemma 4 31B are at 74, our local default Gemma 4 12B at 55, and the small Qwen 4B at 13 before fine-tuning. "
        "Is the local gap about running locally or about size? Gemma 31B and 12B are the same family: the 31B is 19 "
        "points better, so it's mostly size. The orange-blue gap is parts with the right size but the wrong shape, "
        "which the size check alone can't catch."))
    picture_fit(s, fig["models"], 0.79, 2.35, 7.4, 4.35, alt="Right size and right size plus shape per builder model")
    stat(s, 8.75, 2.4, 3.8, "99%", "Claude Sonnet 5.5: right size + shape")
    stat(s, 8.75, 3.85, 3.8, "+19 pts", "Gemma 4 31B vs 12B (same family):\nthe local gap is mostly size")
    text_box(s, 8.75, 5.45, 3.8, 1.1, "Orange vs blue gap: right size, wrong shape. Only the judge catches it.",
             size=14, color=GRAY)

    # 7. Fine-tuning (RQ3)
    s = content_slide(prs, "FINE-TUNING A SMALL LOCAL MODEL", (
        "Can fine-tuning close the gap for a small local model? We trained Qwen 3.5 4B with QLoRA on 1,052 teacher "
        "conversations from Sonnet and Gemini Flash Lite, kept only when the part passed the checks, repairs included. "
        "Every training prompt had three retrieved examples in it, which matters on the next slide. "
        "Right size triples, from 17 to 51 percent, right shape more than doubles, and it needs a third of the tokens "
        "per correct part. But it is still 25 points below the 12B model. On organic shapes it learned to write SDF code "
        "that runs, but not to sculpt the object."))
    data = CategoryChartData()
    data.categories = ["Qwen 4B base", "Qwen 4B fine-tuned", "Gemma 4 12B"]
    data.add_series("Right size", (0.17, 0.51, 0.77))
    data.add_series("Right size + shape", (0.13, 0.30, 0.55))
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.79), Inches(2.35), Inches(7.2),
                               Inches(4.3), data).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size, chart.legend.font.name = Pt(14), FONT
    chart.value_axis.maximum_scale = 1.0
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xD6, 0xD6, 0xD6)
    chart.value_axis.tick_labels.number_format = "0%"
    chart.value_axis.tick_labels.number_format_is_linked = False
    chart.value_axis.format.line.fill.background()
    for ax in (chart.value_axis, chart.category_axis):
        ax.tick_labels.font.size, ax.tick_labels.font.name = Pt(14), FONT
        ax.tick_labels.font.color.rgb = GRAY
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = "0%", False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size, labels.font.bold, labels.font.name = Pt(14), True, FONT
    for series, color in zip(plot.series, (GOLD, INK)):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = color
    stat(s, 8.75, 2.4, 3.8, "1,052", "verified teacher builds\n(QLoRA, 4-bit, on one RTX 3080)")
    stat(s, 8.75, 3.85, 3.8, "85K → 25K", "tokens per correct part")
    text_box(s, 8.75, 5.35, 3.8, 1.1, "Organic: valid mesh 3% → 36%, but right shape only 1%. It learned code that "
             "runs, not the shape.", size=14, color=GRAY)

    # 8. Retrieved examples (RQ2)
    s = content_slide(prs, "DO RETRIEVED EXAMPLES HELP?", (
        "Retrieved examples, the RAG part. For the API models they make no measurable difference to the end result: "
        "Gemini Flash Lite builds the right part 74 percent of the time with or without them, and the Claude models "
        "are within a point. They only change how fast: Gemini's first-try rate goes from 55 to 67 percent, and "
        "without examples the repair loop makes up the gap. The one clear gain is the fine-tuned Qwen, plus 12 points. "
        "But that model only ever saw prompts with three examples during training, so taking them away also takes "
        "away the format it learned. That measures training format as much as retrieval."))
    rows = [("Builder", "RAG", "No RAG", "Diff.", "1st try"),
            ("Claude Sonnet 5.5", "99%", "98%", "+1", "97 / 96"),
            ("Claude Haiku 5.5", "94%", "95%", "−1", "97 / 90"),
            ("Gemini 3.5 Flash Lite", "74%", "74%", "+1", "67 / 55"),
            ("Qwen 4B fine-tuned", "30%", "18%", "+12 *", "28 / 10"),
            ("Qwen 4B base", "13%", "10%", "+3", "10 / 8")]
    widths = [2.75, 1.0, 1.15, 0.95, 1.35]
    tbl = s.shapes.add_table(len(rows), len(widths), Inches(0.79), Inches(2.4), Inches(sum(widths)),
                             Inches(3.3)).table
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = val
            c.fill.solid()
            c.fill.fore_color.rgb = INK if i == 0 else (GOLD if i == 4 else (LIGHT if i % 2 else WHITE))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.name, f.size, f.bold = FONT, Pt(16), i == 0 or i == 4
            f.color.rgb = WHITE if i == 0 else INK
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    text_box(s, 0.79, 5.8, 7.2, 0.8, ["Right size + shape on the 48 main cases, with 3 retrieved examples or none.",
                                      "1st try: correct on the first attempt, RAG / none.  * p = 0.04; the rest is noise."],
             size=12, color=GRAY)
    stat(s, 8.75, 2.4, 3.8, "74% = 74%", "Gemini Flash Lite with and without examples")
    stat(s, 8.75, 3.85, 3.8, "55% → 67%", "its first-try rate: examples help it get\nthere sooner, repair closes the gap")
    card(s, 8.75, 5.3, 3.79, 1.25)
    text_box(s, 8.95, 5.3, 3.4, 1.25, "The fine-tuned Qwen saw 3 examples in every training prompt: its +12 is RAG and "
             "training format together.", size=13, anchor=MSO_ANCHOR.MIDDLE)

    # 9. Repair loop (RQ4)
    s = content_slide(prs, "THE REPAIR LOOP", (
        "Execution feedback. This shows how many runs are correct if we stop after one, two, three or four attempts. "
        "Strong models are right first time, so there's little to fix. Mid-strength models gain the most: Gemini Flash "
        "Lite goes from 67 to 88 percent, Gemma 31B from 52 to 85. On the right, sculpting organic shapes: Gemini's "
        "first attempt gives a valid mesh only 1 percent of the time, after four attempts 82. Sculpting basically only "
        "works because of the loop. The small Qwen models rarely recover from their own script errors."))
    picture_fit(s, fig["attempts"], 0.79, 2.35, 11.75, 3.7, alt="Runs correct after 1 to 4 attempts, per model")
    take = [("67% → 88%", "Gemini Flash Lite"), ("52% → 85%", "Gemma 4 31B"),
            ("1% → 82%", "Gemini, sculpting")]
    for i, (num, label) in enumerate(take):
        x = 0.79 + i * 4.0
        text_box(s, x, 6.15, 1.9, 0.5, num, size=22, bold=True)
        text_box(s, x + 1.95, 6.22, 1.95, 0.4, label, size=13, color=GRAY)

    # 10. Failures and cost
    s = content_slide(prs, "FAILURES AND COST", (
        "What goes wrong? The commonest failure is the right size but the wrong shape, so only the judge catches it: "
        "a feature left out, holes that stop short, a loose request built too literally. Cost: Haiku is cheapest per "
        "correct part at under a tenth of a cent, Sonnet about a cent; Gemma 4 12B runs locally but takes a median "
        "136 seconds per build. The whole round cost about $8.58 at Anthropic and $2.15 at Google."))
    picture_fit(s, fig["failures"], 0.79, 2.35, 7.6, 2.0, left=True,
                alt="Three typical failures: right size, wrong shape")
    picture_fit(s, fig["cost"], 0.79, 4.5, 7.6, 2.3, left=True,
                alt="Right size plus shape against dollars per correct part and build time")
    text_box(s, 8.75, 2.4, 3.8, 0.9, "Right size, wrong shape: a feature left out, holes that stop short, a loose "
             "request built literally", size=14, color=GRAY)
    stat(s, 8.75, 3.45, 3.8, "$0.0009", "per correct part, Haiku\n(Sonnet $0.011, Gemini FL $0.0029)")
    stat(s, 8.75, 5.0, 3.8, "136 s", "median build, Gemma 4 12B locally\n(Gemini FL 3 s, Sonnet 6 s)")

    # 11. Organic (RQ5)
    s = prs.slides.add_slide(layout(prs, "Large Picture with Content - Left"))
    s.shapes.title.text = "SCULPT OR DOWNLOAD?"
    drop_placeholder(s, 13)
    set_text(s.placeholders[1], ["Right shape on 30 organic objects:", "Haiku, sculpted: 82%", "Gemini, sculpted: 52%",
                                 "Thingiverse download: 37%", "Downloads match the name, not always the object"])
    s.notes_slide.notes_text_frame.text = (
        "For organic shapes that code can't easily build, is it better to sculpt with code or download a model? "
        "Sculpting with a strong model wins: Haiku gets the shape right 82 percent of the time, Gemini 52, and the "
        "first matching Thingiverse download 37. The downloads fail in a funny way: the name matches but the object "
        "doesn't. The half avocado is an avocado peeler and the banana is a banana-headed knight. Sometimes a download "
        "wins, like this 3D-scanned parrot.")
    picture_fit(s, fig["organic"], 0.79, 0.79, 7.58, 5.92,
                alt="Half avocado, banana and parrot sculpted by Haiku and Gemini or downloaded from Thingiverse, "
                    "with judge scores")

    # 12. Takeaways
    s = content_slide(prs, "TAKEAWAYS", (
        "To sum up. Weights matter most: model size and fine-tuning move results by 17 to 24 points. Execution "
        "feedback helps mid-strength models most and is what makes sculpting possible. Retrieved examples only help "
        "models that depend on the format. And for organic objects, sculpting with a strong model beats downloading. "
        "Next steps: a harder benchmark, since Sonnet is at the ceiling, evaluating the intake gate, and running the "
        "judge inside the loop so a wrong shape can be repaired like a wrong size."))
    points = [("Weights matter most", "Model size or fine-tuning moves right size + shape by 17–24 points"),
              ("Feedback lifts mid-strength models", "+21 to +33 points from repair; sculpting depends on it"),
              ("Examples help only format-dependent models", "Fine-tuned 4B +12 pts; API models no change"),
              ("Sculpting beats downloading", "Haiku 82% vs Thingiverse 37% on organic shapes")]
    for i, (head, body) in enumerate(points):
        y = 2.4 + i * 0.95
        text_box(s, 0.79, y, 0.6, 0.8, str(i + 1), size=32, bold=True, color=GOLD)
        text_box(s, 1.5, y + 0.02, 6.4, 0.45, head, size=20, bold=True)
        text_box(s, 1.5, y + 0.48, 6.4, 0.45, body, size=14, color=GRAY)
    card(s, 8.4, 2.4, 4.14, 3.75)
    text_box(s, 8.75, 2.7, 3.5, 0.5, "Next", size=20, bold=True)
    text_box(s, 8.75, 3.3, 3.5, 2.7, [[("•  Harder benchmark (Sonnet is at 99%)", {})],
                                        [("•  Evaluate the intake gate and the full app", {})],
                                        [("•  Judge inside the build loop: repair wrong shapes", {})],
                                        [("•  Edit a built part instead of rebuilding", {})]], size=15)

    # 13. Questions / demo
    s = prs.slides.add_slide(layout(prs, "Divider Slide - Black"))
    s.shapes.title.text = "QUESTIONS & DEMO"
    drop_placeholder(s, 1)
    s.notes_slide.notes_text_frame.text = "Switch to the app for the live demo."

    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
