import json
from pathlib import Path

from glyph.tasks import ROOT


def write_events(events, path="events.json"):
    """Dump episode events for the demo Interactions-view (PLAN Demo)."""
    Path(ROOT, path).write_text(json.dumps(events, indent=2))
    return Path(ROOT, path)
