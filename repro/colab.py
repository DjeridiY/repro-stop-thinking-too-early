"""Thin wrapper around google-colab-cli (`colab`): one named VM session."""
import os
import subprocess
import tempfile
from pathlib import Path

COLAB = os.environ.get("COLAB", str(Path.home() / ".local/bin/colab"))


class Session:
    def __init__(self, name: str):
        self.name = name

    def _cli(self, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
        return subprocess.run([COLAB, *args], capture_output=True, text=True, timeout=timeout)

    def start(self, gpu: str = "T4") -> bool:
        return self._cli("new", "-s", self.name, "--gpu", gpu).returncode == 0

    def stop(self) -> None:
        self._cli("stop", "-s", self.name)

    def upload(self, local: Path, remote: str) -> None:
        self._cli("upload", "-s", self.name, str(local), remote).check_returncode()

    def download(self, remote: str, local: Path) -> bool:
        return self._cli("download", "-s", self.name, remote, str(local)).returncode == 0

    def sh(self, command: str, timeout: int = 120) -> str:
        """Run a shell command on the VM through its Python kernel; returns stdout + stderr."""
        code = f"import subprocess\nr = subprocess.run({command!r}, shell=True, capture_output=True, text=True)\nprint(r.stdout + r.stderr)\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            r = self._cli("exec", "-s", self.name, "-f", f.name, "--timeout", str(timeout), timeout=timeout + 60)
        finally:
            os.unlink(f.name)
        return r.stdout + r.stderr
