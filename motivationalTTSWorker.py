"""Run MotivationalTTSModel in the project's uv environment and drive it from any other Python.

Meant for hosts whose own Python cannot install the dependencies, e.g. Google Colab. Importing this
module needs only the standard library and NumPy: `MotivationalTTSWorker` starts the model once in a
background process via `uv run` and exchanges JSON lines with it over stdin/stdout.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import weakref
from pathlib import Path

import numpy as np

LOG_FILE = "motivationalTTSWorker.log"

_workers = weakref.WeakSet()


class MotivationalTTSWorker:
    def __init__(self, **config):
        """
        Start the worker process and block until the model is loaded and compiled.

        Earlier workers in this process are stopped first, so re-running a notebook cell
        does not keep two models on the GPU.

        Args:
            **config: Fields of MotivationalTTSConfig, e.g. seed=None, debug=False.
        """
        for worker in list(_workers):
            worker.close()

        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv not found. Install it with `pip install uv`.")

        self.log_path = Path(LOG_FILE).resolve()
        script = Path(__file__).resolve()
        # Settings meant for the host's Python, like Jupyter's inline matplotlib backend, must not leak into
        # the worker's Python 3.11.
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "MPLBACKEND")}
        with open(self.log_path, "w") as log:
            self._process = subprocess.Popen(
                [uv, "run", "--project", str(script.parent), "python", str(script), json.dumps(config)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                env={**env, "PYTHONUNBUFFERED": "1"},
            )
        # Popen keeps a running child's pipes open after garbage collection; closing stdin ends the worker.
        weakref.finalize(self, self._process.stdin.close)
        _workers.add(self)
        self._request_id = 0

        print(f"Loading and compiling the model in a background process, this takes a few minutes. Log: {self.log_path}")
        try:
            self._receive()
        except BaseException:
            self._process.terminate()
            raise

    def synthesize(self, prompt: str, motivational_factor: float = 0.0):
        """
        Synthesize speech audio from a text prompt and a motivational factor.

        Args:
            prompt (str): The text to be synthesized.
            motivational_factor (float): A value between 0 and 1 representing the motivational intensity.

        Returns:
            tuple: A tuple (audio, sr) where audio is a NumPy array and sr is the sample rate.
        """
        self._request_id += 1
        self._send({"id": self._request_id, "prompt": prompt, "motivational_factor": float(motivational_factor)})

        # Skip replies to earlier calls that were interrupted, e.g. by stopping a notebook cell.
        while (response := self._receive())["id"] != self._request_id:
            if "audio_path" in response:
                os.remove(response["audio_path"])

        if "error" in response:
            raise RuntimeError(f"Synthesis failed in the worker:\n{response['error']}")
        audio = np.load(response["audio_path"])
        os.remove(response["audio_path"])
        return audio, response["sample_rate"]

    def close(self):
        """Stop the worker process, which frees its GPU memory."""
        if self._process.poll() is None:
            self._process.stdin.close()
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait()

    def _send(self, message: dict):
        if self._process.stdin.closed:
            raise RuntimeError("This worker is closed. Create a new MotivationalTTSWorker.")
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            raise self._exited_error() from None

    def _receive(self) -> dict:
        line = self._process.stdout.readline()
        if not line:
            raise self._exited_error()
        return json.loads(line)

    def _exited_error(self) -> RuntimeError:
        returncode = self._process.wait()
        log_tail = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-30:])
        return RuntimeError(f"Worker exited with code {returncode}. Last lines of {self.log_path}:\n{log_tail}")


def _serve(config_json: str):
    # Jupyter interrupts the kernel's whole process group; only the client should react to that.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The model code prints to stdout: keep the real stdout for protocol messages and send fd 1 to the log.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    from motivationalTTS import MotivationalTTSConfig, MotivationalTTSModel

    model = MotivationalTTSModel(MotivationalTTSConfig(**json.loads(config_json)))
    protocol.write(json.dumps({"id": 0}) + "\n")

    for line in sys.stdin:
        request = json.loads(line)
        try:
            audio, sample_rate = model.synthesize(request["prompt"], motivational_factor=request["motivational_factor"])
            fd, audio_path = tempfile.mkstemp(suffix=".npy")
            with os.fdopen(fd, "wb") as f:
                np.save(f, audio)
            response = {"id": request["id"], "audio_path": audio_path, "sample_rate": int(sample_rate)}
        except Exception:
            response = {"id": request["id"], "error": traceback.format_exc()}
        protocol.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    _serve(sys.argv[1])
