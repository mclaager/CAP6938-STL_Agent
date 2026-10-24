# Judge prompts

One file per prompt version, loaded by `llm_judge.py` as the system prompt (file name = version).
Judged result rows record the version, so **never edit a released file**: copy it to a new version instead.

| prompt | written for render | notes |
|---|---|---|
| `v1.txt` | v1: front, right, top + iso front, iso back, from below (angled) | read through-holes as blind pockets in the angled underside view |
| `v2.txt` | v2: front, right, top, bottom (flat) + iso front, iso back (angled) | current default; explains flat vs angled views |

A new render version (`RENDER_VERSION` in `stl_agent/render.py`) needs a prompt that describes its views:
add the file here and its entry to `RENDER_FOR_PROMPT` in `llm_judge.py`.
