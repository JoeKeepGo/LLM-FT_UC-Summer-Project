import os
import subprocess
import sys
import time
from collections import deque

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    print("Missing dependency: rich. Install with: pip install rich")
    raise SystemExit(1) from exc

from llm_ft.experiments import load_student_model_config


STUDENT_MODEL_CONFIG = os.environ.get("LLM_FT_STUDENT_MODEL_CONFIG", "configs/student_model_experiments.json")
COMMON, experiments = load_student_model_config(STUDENT_MODEL_CONFIG)
LOG_BUFFER_LINES = 200

WORKER_ARGS = [
    "model_id",
    "model_alias",
    "train_file",
    "test_file",
    "output_dir",
    "few_shot_count",
    "max_new_tokens",
    "temperature",
    "do_sample",
    "load_in_8bit",
    "load_in_4bit",
    "seed_base",
    "num_runs",
    "eval_batch_size",
    "max_seq_length",
    "train_batch_size",
    "grad_accumulation",
    "learning_rate",
    "max_steps",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
]


def _format_duration(seconds):
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def _format_value(value, default="n/a"):
    if value is None:
        return default
    return str(value)


def _add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


class HeaderView:
    def __init__(self):
        self.exp = None
        self.idx = 0
        self.total = 0
        self.total_start = time.time()
        self.exp_start = None
        self.status = "idle"
        self.log_file = None

    def set_experiment(self, exp, idx, total, log_file):
        self.exp = exp
        self.idx = idx
        self.total = total
        self.log_file = log_file
        self.exp_start = time.time()
        self.status = "running"

    def set_status(self, status):
        self.status = status

    def __rich__(self):
        if not self.exp:
            return Panel("Waiting for student model experiments...", title="Student Model Status", border_style="blue")

        exp = self.exp
        remaining = max(0, self.total - self.idx)
        now = time.time()
        total_elapsed = _format_duration(now - self.total_start)
        exp_elapsed = _format_duration(now - (self.exp_start or now))

        status_color = "white"
        if self.status.startswith("running"):
            status_color = "yellow"
        elif self.status.startswith("success"):
            status_color = "green"
        elif self.status.startswith("failed"):
            status_color = "red"

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Experiment:", f"[{self.idx}/{self.total}] {exp.get('run_name', 'unknown')}")
        table.add_row("Mode:", _format_value(exp.get("mode")))
        table.add_row("Model:", f"{_format_value(exp.get('model_alias'))} | {_format_value(exp.get('model_id'))}")
        table.add_row(
            "Data:",
            f"train={_format_value(exp.get('train_file'))} | test={_format_value(exp.get('test_file'))}",
        )
        table.add_row(
            "Run:",
            f"runs={_format_value(exp.get('num_runs'))} | output={_format_value(exp.get('output_dir'))}",
        )
        table.add_row(
            "Progress:",
            f"remaining={remaining} | exp_elapsed={exp_elapsed} | total_elapsed={total_elapsed} | status=[{status_color}]{self.status}[/{status_color}]",
        )
        if self.log_file:
            table.add_row("Log File:", self.log_file)
        return Panel(table, title="Student Model Status", border_style="blue")


class LogView:
    def __init__(self, max_lines=200):
        self.lines = deque(maxlen=max_lines)

    def append(self, line):
        if line is not None:
            self.lines.append(line)

    def clear(self):
        self.lines.clear()

    def __rich_console__(self, console, options):
        if not self.lines:
            yield Panel(Text("Waiting for output..."), title="Live Output", border_style="cyan")
            return
        lines = list(self.lines)
        if options.max_height is not None:
            interior = max(1, options.max_height - 2)
            lines = lines[-interior:]
        yield Panel(Text("\n".join(lines)), title="Live Output", border_style="cyan")


def run_experiments():
    console = Console()
    python_executable = sys.executable
    script_path = "student_model_worker.py"

    if not os.path.exists(script_path):
        console.print(f"[red]Error: Could not find {script_path}[/red]")
        return
    if not experiments:
        console.print("[red]No student model experiments found. Check the JSON config.[/red]")
        return

    header = HeaderView()
    log_view = LogView(LOG_BUFFER_LINES)
    layout = Layout()
    layout.split(Layout(name="header", size=11), Layout(name="body"))
    layout["header"].update(header)
    layout["body"].update(log_view)

    total = len(experiments)
    header.total_start = time.time()
    log_view.append(f"Starting {total} student model experiments...")

    with Live(layout, console=console, refresh_per_second=4, screen=True):
        for index, exp in enumerate(experiments, start=1):
            log_view.clear()
            log_file = f"log_student_{exp['run_name']}.txt"
            header.set_experiment(exp, index, total, log_file)

            cmd = [python_executable, script_path, "--run_name", exp["run_name"], "--mode", exp["mode"]]
            for key in WORKER_ARGS:
                _add_arg(cmd, f"--{key}", exp.get(key))

            log_view.append(f"Experiment [{index}/{total}]: {exp['run_name']}")
            log_view.append(f"Config: {exp}")
            log_view.append(f"Logs: {log_file}")
            log_view.append(f"Command: {' '.join(cmd)}")

            start_time = time.time()
            return_code = 1
            try:
                with open(log_file, "w", encoding="utf-8") as handle:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    if process.stdout is None:
                        raise RuntimeError("Failed to capture process output.")
                    for line in process.stdout:
                        handle.write(line)
                        handle.flush()
                        log_view.append(line.rstrip("\n"))
                    return_code = process.wait()
            except Exception as exc:
                log_view.append(f"Controller error: {exc}")

            duration = time.time() - start_time
            if return_code == 0:
                header.set_status(f"success ({duration/60:.2f}m)")
                log_view.append(f"Success. Duration: {duration/60:.2f} mins")
            else:
                header.set_status(f"failed (code={return_code})")
                log_view.append(f"Failed. Check {log_file} for details.")

            time.sleep(5)


if __name__ == "__main__":
    run_experiments()
