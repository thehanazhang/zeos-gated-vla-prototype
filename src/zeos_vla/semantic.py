"""Strict, bounded semantic proposal: no embeddings and no executable model text."""

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class SemanticCommand:
    action: str
    target: str
    speed: str

    def json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def parse(cls, text):
        if not isinstance(text, str) or len(text) > 1024:
            raise ValueError("Command must be bounded JSON text")

        def unique(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError("Duplicate command field")
                obj[k] = v
            return obj

        obj = json.loads(text, object_pairs_hook=unique)
        if not isinstance(obj, dict) or set(obj) != {"action", "target", "speed"}:
            raise ValueError("Unexpected semantic schema")
        if (
            obj["action"] != "manipulate"
            or obj["target"] != "package"
            or obj["speed"] not in ("slow", "fast")
        ):
            raise ValueError("Unsupported command")
        return cls(**obj)
