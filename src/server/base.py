import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from abc import abstractmethod

import requests


def _stream_output(pipe, lines: list[str]) -> None:
	"""Read lines from a pipe, print them, and store in a list."""
	for raw in iter(pipe.readline, b""):
		line = raw.decode("utf-8", errors="replace").rstrip("\n")
		print(f"  [server] {line}", flush=True)
		lines.append(line)
	pipe.close()


@dataclass
class InferenceServer:
	"""Base class for managing a local inference server process."""

	model: str
	host: str = "127.0.0.1"
	port: int = 30000
	tp: int = 1
	mem_fraction: float = 0.9
	served_model_name: str | None = None
	gpus: str | None = None
	extra_args: list[str] = field(default_factory=list)

	_process: subprocess.Popen | None = field(default=None, init=False, repr=False)
	_output_lines: list[str] = field(default_factory=list, init=False, repr=False)

	@property
	def base_url(self) -> str:
		return f"http://{self.host}:{self.port}/v1"

	@abstractmethod
	def _build_cmd(self) -> list[str]:
		...

	def is_healthy(self) -> bool:
		"""Check if the server is responding."""
		try:
			resp = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
			return resp.status_code == 200
		except (requests.ConnectionError, requests.Timeout):
			return False

	def launch(self, timeout: int = 300) -> None:
		"""Start the server and wait until it's healthy."""
		cmd = self._build_cmd()
		env = None
		if self.gpus is not None:
			gpus = ",".join(str(g) for g in self.gpus) if isinstance(self.gpus, (list, tuple)) else str(self.gpus)
			env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
		self._output_lines = []
		self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
		reader = threading.Thread(target=_stream_output, args=(self._process.stdout, self._output_lines), daemon=True)
		reader.start()

		deadline = time.time() + timeout
		while time.time() < deadline:
			if self._process.poll() is not None:
				reader.join(timeout=5)
				raise RuntimeError(
					f"{self.__class__.__name__} exited with code {self._process.returncode}\n"
					+ "\n".join(self._output_lines[-50:])
				)
			if self.is_healthy():
				return
			time.sleep(2)
		self.stop()
		raise TimeoutError(
			f"{self.__class__.__name__} did not become healthy within {timeout}s\n"
			f"Last output:\n" + "\n".join(self._output_lines[-50:])
		)

	def stop(self) -> None:
		"""Terminate the server process."""
		if self._process is None:
			return
		self._process.terminate()
		try:
			self._process.wait(timeout=30)
		except subprocess.TimeoutExpired:
			self._process.kill()
			self._process.wait(timeout=10)
		self._process = None
