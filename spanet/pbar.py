from collections import OrderedDict
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar, RichProgressBarTheme, MetricsTextColumn, CustomProgress
)
from pytorch_lightning.trainer import trainer as pltrainer
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

# It was initially intended to be a table - might come in the future
class MyMetricsTable:
    def __init__(self):
        self._metrics = None
        self._ordered_metrics = None

    def update(self, metrics):
        self._metrics = metrics

    def _format_metrics(self):
        self._metrics.pop("v_num")
        self._ordered_metrics = OrderedDict()
        self._ordered_metrics["epoch"] = self._metrics["epoch"]
        self._ordered_metrics["avg_jet_acc"] = round(self._metrics["validation_average_jet_accuracy"], 3)
        for key, value in self._metrics.items():
            if key not in ["epoch", "validation_average_jet_accuracy"]:
                self._ordered_metrics[key] = round(value, 3)

    def render(self):
        self._format_metrics()
        return dict(self._ordered_metrics)


class MyProgressBar(RichProgressBar):
    def __init__(self):
        self._metrics_table = MyMetricsTable()
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
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def _update_metrics(self, trainer, pl_module) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        if self._metric_component:
            self._metric_component.update(metrics)
            self._metrics_table.update(metrics | {"epoch": trainer.current_epoch})

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)
            self.progress.console.print(
                self._metrics_table.render()
            )
        self.reset_dataloader_idx_tracker()
