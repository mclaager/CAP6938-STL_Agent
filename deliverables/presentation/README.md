# Presentation (7 minutes)

`stl_agent_talk.pptx`: 11 slides plus one backup slide, on the UCF brand template
(`UCF-PowerPoint-Template-...potx`). Speaker notes on every slide hold the talking points (~830 words,
about 6.5 minutes). The live demo follows the "Questions & demo" slide.

Every number comes from the report (`../report/sec/5_results.tex`), which takes them from
`eval/results/report-v1.md`. Figures are the report's own, rendered from `../report/figures/*.pdf`.

Rebuild after changing `build_slides.py` or the report figures (needs python-pptx, which is not a pixi
dependency, and `pdftoppm` from Poppler):

```
python -m venv /tmp/slides-venv && /tmp/slides-venv/bin/pip install python-pptx
/tmp/slides-venv/bin/python deliverables/presentation/build_slides.py
```

Edits made by hand in PowerPoint are lost on a rebuild: either edit the script, or stop rebuilding.
