import os
import subprocess
import sys
from pathlib import Path

separator = sys.argv.index("--")
tool = sys.argv[1]
tool_args = sys.argv[2:separator]
files = sys.argv[separator + 1 :]

repository = Path(__file__).resolve().parent.parent
desktop = repository / "desktop"
executable = desktop / "node_modules" / ".bin" / tool
if os.name == "nt":
    executable = executable.with_suffix(".cmd")

desktop_files = [str((repository / file).resolve().relative_to(desktop)) for file in files]
subprocess.run([str(executable), *tool_args, *desktop_files], cwd=desktop, check=True)
