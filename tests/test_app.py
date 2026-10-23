"""Chat flow through the app's handlers with the gate and build stubbed (no LLM calls)."""

from stl_agent import app
from stl_agent.gate import decide_from_report

CRITICAL = {"scope": "in_scope", "spec": "stand",
            "unknowns": [{"param": "width", "default": "75", "critical": True, "question": "Phone width?"}]}
READY = {"scope": "in_scope", "spec": "stand 80mm", "name": "phone-stand",
         "unknowns": [{"param": "angle", "default": "60", "critical": False}]}


def last(gen):
    *_, final = gen
    return final


def test_clarify_then_build(monkeypatch, tmp_path):
    calls = []

    def fake_gate(conversation, rounds):
        calls.append(rounds)
        return decide_from_report(CRITICAL if rounds == 0 else READY, "x", rounds)

    def fake_pipeline(spec, log, name=None, **kw):
        log("working on " + spec)
        return str(tmp_path / f"{name}.stl"), "code", {"success": True}

    monkeypatch.setattr(app, "run_gate", fake_gate)
    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    state = app.new_state()

    chat = last(app.on_message("phone stand", [], state, False))[1]
    assert "Phone width?" in chat[-1]["content"]
    assert state["rounds"] == 1 and state["spec"] is None

    chat = last(app.on_message("80mm", chat, state, False))[1]
    assert "stand 80mm" in chat[-1]["content"]
    assert state["spec"] == "stand 80mm" and state["name"] == "phone-stand"

    chat, state, _, log = last(app.on_build(chat, state))
    assert "phone-stand.stl" in chat[-1]["content"]
    assert "working on stand 80mm" in log
    assert state["result"]["stl"].endswith("phone-stand.stl")
    assert state["spec"] is None and state["rounds"] == 0
    assert calls == [0, 1]
