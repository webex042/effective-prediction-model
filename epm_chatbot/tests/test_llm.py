"""The optional LLM may only rephrase grounded facts; fabricated numbers are rejected."""

from epm_chatbot.backend.llm import LLMService, ungrounded_numbers
from epm_chatbot.config import Settings


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, system, user, max_tokens=400):
        self.calls += 1
        return self.reply

    def stream(self, system, user, max_tokens=400):
        for part in self.reply.split(" "):
            yield part + " "


FACTS = {"prediction": "species = setosa", "probability": 0.9782, "heldout": {"accuracy": 0.933333}}


def test_grounding_guard():
    assert ungrounded_numbers("Accuracy was 0.93 (93.3%) with probability 97.8%.", FACTS) == []
    assert ungrounded_numbers("It is 99% certain and accuracy 0.95.", FACTS) == ["99%", "0.95"]


def test_disabled_without_configuration():
    svc = LLMService(Settings())
    assert not svc.enabled
    out = svc.rephrase(FACTS, "fallback text")
    assert out == {"text": "fallback text", "source": "deterministic"}
    assert svc.classify_intent("anything", ["predict"]) is None


def test_rephrase_accepts_grounded_and_streams():
    svc = LLMService(Settings(), client=FakeClient("The model chose setosa with probability 97.8% and accuracy 0.93."))
    deltas = []
    out = svc.rephrase(FACTS, "fallback", on_delta=deltas.append)
    assert out["source"] == "llm" and "setosa" in out["text"] and len(deltas) > 3


def test_rephrase_rejects_fabricated_numbers():
    svc = LLMService(Settings(), client=FakeClient("The model is 99.9% accurate on 5000 patients."))
    out = svc.rephrase(FACTS, "fallback")
    assert out["source"] == "deterministic" and out["text"] == "fallback" and out["rejected"]


def test_intent_classification_is_constrained():
    assert LLMService(Settings(), client=FakeClient("predict")).classify_intent("hm", ["predict", "help"]) == "predict"
    assert LLMService(Settings(), client=FakeClient("delete_everything")).classify_intent("x", ["predict"]) is None
