from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar, RichProgressBarTheme, MetricsTextColumn, CustomProgress
)
from rich import get_console, reconfigure
from rich.text import Text
from rich.progress import TaskID
from typing import cast

class MyMetricsTextColumn(MetricsTextColumn):
    def __init__(
            self, trainer, style, text_delimiter, metrics_format
    ):
        super().__init__(trainer, style, text_delimiter, metrics_format)

    def render(self, task):
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
                self._trainer.state.fn != "fit"
                or self._trainer.sanity_checking
                or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        metrics_texts = self._generate_metrics_texts()
        text = self._text_delimiter.join(metrics_texts)
        return Text(text, justify="left", style="bold cyan")


class MyProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__()

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MyMetricsTextColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False