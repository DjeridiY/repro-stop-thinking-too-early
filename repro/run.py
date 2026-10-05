"""Run the next unfinished seed on a free Colab T4, keep the kernel busy, fetch the outputs, release the VM.

    python -m repro.run          # safe to call repeatedly (timer): exits if a run is active or all seeds are done
"""
import fcntl
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .colab import Session

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)
VM_DIR = "/content/w/upstream"
POLL_S, MAX_POLLS = 240, 90  # 6 h ceiling; a full seed takes ~3.5 h on a T4


def tag(seed: int) -> str:
    return "o14_a6" if seed == 0 else f"o14_a6_s{seed}"


def authors_args(seed: int) -> str:
    """Arguments of e71_ouro_map.py exactly as in the authors' REPRODUCE.md."""
    args = f"--a 6 --tag {tag(seed)}"
    return args if seed == 0 else f"{args} --seed {seed} --Ts_eval 1,2,3,4,6,8"


def outputs(seed: int) -> list[str]:
    return ["train.log", "environment.txt", f"e71_ouro_{tag(seed)}.json", f"e71_map_{tag(seed)}.pt"]


class Run:
    def __init__(self, seed: int):
        self.seed, self.dir = seed, RESULTS / f"seed{seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vm = Session("repro-full")

    def log(self, msg: str) -> None:
        with open(self.dir / "runner.log", "a") as f:
            f.write(f"{datetime.now(timezone.utc):%FT%TZ} {msg.strip()}\n")

    def fetch(self, *names: str) -> None:
        for name in names:
            self.vm.download(f"{VM_DIR}/results/{name}", self.dir / name)

    def launch(self) -> bool:
        if not self.vm.start():
            self.log("no T4 available (quota?), retry later")
            return False
        bundle = ROOT / ".bundle.tar"
        with tarfile.open(bundle, "w") as tar:
            tar.add(ROOT / "upstream", "upstream", filter=lambda t: None if t.name.endswith(".git") else t)
            tar.add(ROOT / "patches", "patches")
            tar.add(ROOT / "repro" / "vm_setup.sh", "vm_setup.sh")
        self.vm.upload(bundle, "/content/bundle.tar")
        self.log(self.vm.sh(f"mkdir -p /content/w && tar xf /content/bundle.tar -C /content/w "
                            f"&& bash /content/w/vm_setup.sh '{authors_args(self.seed)}'", timeout=900))
        return True

    def status(self) -> str:
        return self.vm.sh(f"grep -E '^step|^c=|^done|Traceback' {VM_DIR}/results/train.log | tail -1; "
                          "echo procs=$(pgrep -fc '[e]71_ouro_map')", timeout=60)

    def watch(self) -> None:
        for _ in range(MAX_POLLS):
            time.sleep(POLL_S)  # each poll also keeps the Colab kernel from being reclaimed as idle
            s = self.status()
            self.log(s.replace("\n", " "))
            self.fetch("train.log", f"e71_ouro_{tag(self.seed)}.json")
            if "not found" in s or "procs=0" in s or "\ndone" in f"\n{s}" or "Traceback" in s:
                return

    def __call__(self) -> None:
        self.log(f"seed {self.seed}: start")
        if not self.launch():
            return
        try:
            self.watch()
            self.fetch(*outputs(self.seed))
        finally:
            self.vm.stop()
        finished = "\ndone " in (self.dir / "train.log").read_text() if (self.dir / "train.log").exists() else False
        if finished:
            (self.dir / "DONE").touch()
        self.log(f"seed {self.seed}: {'done' if finished else 'incomplete, will retry'}")


def main() -> None:
    with open(RESULTS / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        todo = [s for s in SEEDS if not (RESULTS / f"seed{s}" / "DONE").exists()]
        if todo:
            Run(todo[0])()


if __name__ == "__main__":
    main()
