from collections import OrderedDict
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar, RichProgressBarTheme, MetricsTextColumn, CustomProgress
)
from pytorch_lightning.trainer import trainer as pltrainer
from rich import get_console, reconfigure
from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich.progress import TaskID
from typing import cast, Iterable


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
    def __init__(self, central_metric):
        self._central_metric_name, self._central_metric_mode = central_metric
        self._metrics = None
        self._ordered_metrics = None
        self._best_values = [0, 0, 0]
        self._istop3 = False


    def update_metrics(self, metrics):
        self._metrics = metrics

    def _update_best_values(self):
        if self._central_metric_mode == 'max':
            if self._metrics[self._central_metric_name] > self._best_values[-1]:
                self._best_values.append(self._metrics[self._central_metric_name])
                self._best_values.sort(reverse = True)
                self._best_values.pop(-1)
                self._istop3 = True
            else:
                self._istop3 = False
        else:
            if self._metrics[self._central_metric_name] < self._best_values[-1]:
                self._best_values.append(self._metrics[self._central_metric_name])
                self._best_values.sort()
                self._best_values.pop(-1)
                self._istop3 = True
            else:
                self._istop3 = False

    def _format_metrics(self):
        self._metrics.pop("v_num")
        self._ordered_metrics = OrderedDict()
        self._ordered_metrics["epoch"] = self._metrics["epoch"]
        self._ordered_metrics[self._central_metric_name] = round(self._metrics[self._central_metric_name], 3)
        for key, value in self._metrics.items():
            if key not in ["epoch", self._central_metric_name]:
                self._ordered_metrics[key] = round(value, 3)
        self._ordered_metrics["is_top3"] = self._istop3

    def render(self):
        self._update_best_values()
        self._format_metrics()
        return dict(self._ordered_metrics)



class MyProgressBar(RichProgressBar):
    def __init__(self, central_metric):
        self._central_metric = central_metric
        #self._metrics_table = MyMetricsTable(self._central_metric)
        super().__init__()

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self.progress = MyCustomProgress(
                *self.configure_columns(trainer),
                self._central_metric,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.state.fn == "fit":
            self.progress.update_metrics(self.get_metrics(trainer, pl_module) | {"epoch": trainer.current_epoch})
            self.progress.update_table()
        self.reset_dataloader_idx_tracker()


# Modify the already existing PyTorch progress class
class MyCustomProgress(CustomProgress):
    def __init__(self, *args, **kwargs):
        # central metric is last argument
        self._central_metric_name, self._central_metric_mode = args[-1]
        self._metrics = []
        self._best_values = [0, 0, 0]
        self._istop3 = False
        self.table = Table(title = "Metrics")

        super().__init__(*args[:-1], **kwargs)

    def update_metrics(self, metrics):
        self._update_best_values(metrics)
        self._metrics.append(
            self._format_metrics(metrics)
        )

    def update_table(self):
        table = Table(title = "Metrics")
        for item in self._metrics[0].keys():
            table.add_column(f"{item}", header_style = "bold magenta", justify = "right")
        for epoch in self._metrics:
            if len(epoch) == 1: epoch["epoch"] = -1
            istop3_color = ["italic red", "italic green"][epoch["is_top3"]]
            table.add_row(
                "[magenta]%s[/magenta]" %epoch["epoch"],
                f"[bold cyan]{epoch[self._central_metric_name]:.4f}[/bold cyan]",
                *[f"[cyan]{value:.4f}[/cyan]" for value in epoch.values()][2:-1],
                "[%s]%s[/%s]" %(istop3_color, epoch["is_top3"], istop3_color)
            )

        self.table = table

    def _update_best_values(self, metrics):
        if self._central_metric_mode == 'max':
            if metrics[self._central_metric_name] > self._best_values[-1]:
                self._best_values.append(metrics[self._central_metric_name])
                self._best_values.sort(reverse = True)
                self._best_values.pop(-1)
                self._istop3 = True
            else:
                self._istop3 = False
        else:
            if metrics[self._central_metric_name] < self._best_values[-1]:
                self._best_values.append(metrics[self._central_metric_name])
                self._best_values.sort()
                self._best_values.pop(-1)
                self._istop3 = True
            else:
                self._istop3 = False

    # format the metric dict to fit the table design
    def _format_metrics(self, metrics):
        metrics_formatted = OrderedDict()
        metrics_formatted["epoch"] = metrics["epoch"]
        metrics_formatted[self._central_metric_name] = metrics[self._central_metric_name]
        for key, value in metrics.items():
            if key not in ["epoch", self._central_metric_name, "v_num"]:
                metrics_formatted[key] = value
        metrics_formatted["is_top3"] = self._istop3
        return metrics_formatted

    def get_renderables(self):
        renderables = super().get_renderables()
        renderables = Group(self.table, *renderables)
        yield renderables