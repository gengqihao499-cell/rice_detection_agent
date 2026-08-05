from __future__ import annotations

import argparse
import json

from rice_agent.services.detector import RiceDiseaseDetector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    result = RiceDiseaseDetector().detect(
        image_path=args.image,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
