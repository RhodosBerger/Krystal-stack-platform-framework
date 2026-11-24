"""Standard cargo-zigbuild style wrapper invoking `zig cc -target x86_64-windows-gnu` for Cargo."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    zig = shutil.which("zig") or shutil.which("zig.exe")
    if not zig:
        candidate = (
            Path.home()
            / "AppData/Local/Microsoft/WinGet/Packages/zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "zig-x86_64-windows-0.17.0/zig.exe"
        )
        if candidate.exists():
            zig = str(candidate)
        else:
            sys.stderr.write("zig.exe not found\n")
            return 1

    filtered: list[str] = [
        zig,
        "cc",
        "-target",
        "x86_64-windows-gnu",
        "-L",
        r"C:\Windows\System32",
    ]
    for arg in sys.argv[1:]:
        # `rustc` passes both `symbols.o` (COFF .drectve export table) and `-Wl,<path>.def`;
        # Zig's internal LLD consumes `symbols.o` directly and rejects `-Wl,*.def`.
        if arg.startswith("-Wl,") and arg.lower().endswith(".def"):
            continue
        if arg in (
            "-nolibc",
            "-Wl,-Bstatic",
            "-Wl,-Bdynamic",
            "-nodefaultlibs",
            "-Wl,-O1",
        ):
            continue
        if arg == "--unwindlib=libunwind":
            filtered.append("-lunwind")
            continue
        if arg.startswith("-Wl,--out-implib="):
            continue
        if arg == "-lsynchronization":
            filtered.append("-lapi-ms-win-core-synch-l1-2-0")
            continue
        filtered.append(arg)

    return subprocess.call(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
