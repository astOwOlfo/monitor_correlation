import sys
from dataclasses import dataclass

from src.server.base import InferenceServer


@dataclass
class VLLMServer(InferenceServer):
	"""Manages a local vLLM OpenAI-compatible server process."""

	port: int = 8000

	def _build_cmd(self) -> list[str]:
		cmd = [
			sys.executable, "-m", "vllm.entrypoints.openai.api_server",
			"--model", self.model,
			"--host", self.host,
			"--port", str(self.port),
			"--tensor-parallel-size", str(self.tp),
			"--gpu-memory-utilization", str(self.mem_fraction),
		]
		if self.served_model_name:
			cmd += ["--served-model-name", self.served_model_name]
		return cmd + self.extra_args
