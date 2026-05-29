import sys
from dataclasses import dataclass

from src.server.base import InferenceServer


@dataclass
class SGLangServer(InferenceServer):
	"""Manages a local SGLang OpenAI-compatible server process."""

	port: int = 30000

	def _build_cmd(self) -> list[str]:
		cmd = [
			sys.executable, "-m", "sglang.launch_server",
			"--model-path", self.model,
			"--host", self.host,
			"--port", str(self.port),
			"--tp", str(self.tp),
			"--mem-fraction-static", str(self.mem_fraction),
		]
		if self.served_model_name:
			cmd += ["--served-model-name", self.served_model_name]
		return cmd + self.extra_args
