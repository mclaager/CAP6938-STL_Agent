# Presentation (7 minutes)

`stl_agent_talk.pptx`: 13 slides on the UCF brand template
(see "Template" below). Speaker notes on every slide hold the talking points (~1,000 words,
about 7.5 minutes; 8-9 is fine). The live demo follows the "Questions & demo" slide.

Every number comes from the report (`../report/sec/5_results.tex`), which takes them from
`eval/results/report-v1.md`. Figures are the report's own, rendered from `../report/figures/*.pdf`.

## Template

The UCF template is not in the repo (it is a UCF brand asset). To rebuild the deck, download it from UCF's brand
site into this folder, keeping its file name:

```
curl -L -o deliverables/presentation/UCF-PowerPoint-Template-Brand2022-Helvetica-9-22-22-FinalV2.potx \
  https://www.ucf.edu/brand/wp-content/blogs.dir/13/files/2022/09/UCF-PowerPoint-Template-Brand2022-Helvetica-9-22-22-FinalV2.potx
```

The built deck, `stl_agent_talk.pptx`, is committed and opens without it.

## Rebuild

Rebuild after changing `build_slides.py` or the report figures (needs python-pptx, which is not a pixi
dependency, and `pdftoppm` from Poppler):

```
python -m venv /tmp/slides-venv && /tmp/slides-venv/bin/pip install python-pptx
/tmp/slides-venv/bin/python deliverables/presentation/build_slides.py
```

Edits made by hand in PowerPoint are lost on a rebuild: either edit the script, or stop rebuilding.
