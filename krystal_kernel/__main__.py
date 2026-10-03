"""CLI: python -m krystal_kernel [profile|calibrate|params|bench]"""
import json
import sys

from . import calibrate, derive_params, detect_profile


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "profile"
    if cmd == "profile":
        print(json.dumps(detect_profile().to_dict(), indent=2))
    elif cmd == "calibrate":
        print(json.dumps(calibrate(), indent=2))
    elif cmd == "params":
        print(json.dumps(derive_params(detect_profile()).explain(), indent=2, default=str))
    else:
        print("usage: python -m krystal_kernel [profile|calibrate|params]")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
