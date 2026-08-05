from __future__ import annotations

import json

from rice_agent.services.detector import RiceDiseaseDetector


if __name__ == "__main__":
    detector = RiceDiseaseDetector()
    print(json.dumps(
        detector.model_info(),
        ensure_ascii=False,
        indent=2,
    ))
