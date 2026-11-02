"""Composite-only terminal reporting for ordered task stages."""


class CompositeStageReporterMixin:
    """Expose concise stage-transition output only to composite tasks."""

    PRINT_STAGE_MESSAGES = True
    _GREEN = "\033[92m"
    _RESET = "\033[0m"

    def _report_stage_complete(
        self,
        completed_index,
        total_stages,
        next_stage=None,
    ):
        if not self.PRINT_STAGE_MESSAGES:
            return
        completed_index = int(completed_index)
        total_stages = int(total_stages)
        if not 1 <= completed_index <= total_stages:
            raise ValueError("completed stage index must be within the stage count")
        message = f"stage {completed_index}/{total_stages}"
        if next_stage is not None:
            message += f", next stage: {next_stage}"
        print(f"{self._GREEN}{message}{self._RESET}", flush=True)


__all__ = ["CompositeStageReporterMixin"]
