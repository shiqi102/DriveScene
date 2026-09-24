import argparse
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from drivescene.upstream import register

parser = argparse.ArgumentParser()
parser.add_argument("--detection-only", action="store_true")
args = parser.parse_args()
modules = register(root, not args.detection_only, not args.detection_only)
print(json.dumps({"registered": modules}, indent=2))
