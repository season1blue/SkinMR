try:
	from .model import LlavaLlamaForCausalLM
except Exception:
	# Keep package importable for serve utilities that do not require model class at import time.
	pass
