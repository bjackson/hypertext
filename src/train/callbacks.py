"""Training callbacks for logging and checkpointing."""

from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
	"""Custom callback for additional logging."""

	def on_log(self, args, state, control, logs=None, **kwargs):
		if logs:
			logger.info(f"Step {state.global_step}: {logs}")
