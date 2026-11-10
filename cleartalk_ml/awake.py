"""Ask the operating system not to idle-sleep while a long job runs.

No setting is changed: the request lasts only as long as the calling process and is the same
mechanism a video player uses. A closed lid or a manual Sleep still sleeps the computer.
"""

import contextlib
import os
import subprocess
import sys


def keep_awake() -> None:
    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            import ctypes

            continuous, system_required = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required)
        elif sys.platform == "darwin":
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
