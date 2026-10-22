"""EPM ML chatbot — a natural-language interface over the dataset registry, algorithm
catalogue and benchmark runner. Rule-based + TF-IDF intent recognition; no LLM, no
network calls, no arbitrary code execution."""

from src.chatbot.engine import ChatEngine, ChatError, Session
from src.chatbot.intents import INTENTS, classify_intent, parse_intent
from src.chatbot.responses import ChatResponse
from src.chatbot.slots import ChatSlots, extract_slots

__all__ = ["ChatEngine", "ChatError", "Session", "ChatResponse", "ChatSlots", "INTENTS",
           "classify_intent", "parse_intent", "extract_slots"]
