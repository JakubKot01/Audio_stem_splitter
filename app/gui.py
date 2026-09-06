from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
import importlib.util
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.model_catalog import MODEL_FILENAMES, get_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_AUDIO_TYPES = [
    ("Audio files", "*.wav *.mp3 *.flac"),
    ("WAV files", "*.wav"),
    ("MP3 files", "*.mp3"),
    ("FLAC files", "*.flac"),
    ("All files", "*.*"),
]
MODELS = MODEL_FILENAMES
MDX_PYTHON = PROJECT_ROOT / ".venv-mdx" / "Scripts" / "python.exe"


def build_separation_command(
    python_executable: str,
    input_file: str,
    output_dir: str,
    experiments_dir: str,
    model_name: str,
    device: str,
    segment: int,
    shifts: int,
    mp3: bool,
    run_name: str,
    reference_dir: str,
    evaluation_mapping: str,
) -> list[str]:
    """Build the existing CLI command used by the desktop interface."""
    command = [
        python_executable,
        "-m",
        "app.main",
        input_file,
        "--output-dir",
        output_dir,
        "--experiments-dir",
        experiments_dir,
        "--model-name",
        model_name,
        "--shifts",
        str(shifts),
    ]

    if device != "auto":
        command.extend(["--device", device])
    if segment > 0:
        command.extend(["--segment", str(segment)])
    if mp3:
        command.append("--mp3")
    if run_name:
        command.extend(["--run-name", run_name])
    if reference_dir:
        command.extend(["--reference-dir", reference_dir])
    if evaluation_mapping:
        command.extend(["--evaluation-mapping", evaluation_mapping])

    return command


def build_audio_separator_command(
    python_executable: str,
    input_file: str,
    output_dir: str,
    experiments_dir: str,
    model_name: str,
    mp3: bool,
    run_name: str,
    reference_dir: str,
    evaluation_mapping: str,
) -> list[str]:
    """Build the MDX/MDXC command executed inside the optional MDX venv."""
    command = [
        python_executable,
        "-m",
        "app.audio_separator_main",
        input_file,
        "--output-dir",
        output_dir,
        "--experiments-dir",
        experiments_dir,
        "--model-name",
        model_name,
        "--model-file-dir",
        str(PROJECT_ROOT / "models" / "audio-separator"),
    ]
    if mp3:
        command.append("--mp3")
    if run_name:
        command.extend(["--run-name", run_name])
    if reference_dir:
        command.extend(["--reference-dir", reference_dir])
    if evaluation_mapping:
        command.extend(["--evaluation-mapping", evaluation_mapping])
    return command


def check_audio_separator_environment(python_executable: Path) -> str | None:
    """Return a user-facing problem description, or None when MDX is ready."""
    if not python_executable.is_file():
        return f"Python executable was not found: {python_executable}"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import importlib.util,sys; "
                "sys.exit(0 if importlib.util.find_spec('audio_separator') else 2)",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"Could not start the MDX Python environment: {error}"
    if result.returncode == 0:
        return None
    details = (result.stderr or result.stdout).strip()
    if details:
        return details
    return "The audio-separator package is not installed in the selected environment."


class StemSplitterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Audio Stem Splitter")
        self.root.geometry("920x720")
        self.root.minsize(760, 620)

        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.stop_requested = False

        self.input_file = tk.StringVar()
        self.model_name = tk.StringVar(value=MODELS[0])
        self.shifts = tk.IntVar(value=1)
        self.device = tk.StringVar(value="auto")
        self.segment = tk.IntVar(value=0)
        self.mp3 = tk.BooleanVar(value=False)
        self.run_name = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(PROJECT_ROOT / "outputs"))
        self.experiments_dir = tk.StringVar(value=str(PROJECT_ROOT / "experiments"))
        self.reference_dir = tk.StringVar()
        self.mapping_file = tk.StringVar()
        self.model_description = tk.StringVar()
        self.status = tk.StringVar(value="Ready")

        self._configure_style()
        self._build_layout()
        self.model_name.trace_add("write", self._model_changed)
        self.reference_dir.trace_add("write", self._reference_changed)
        self._model_changed()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_messages)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", foreground="#555555")
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(10, weight=1)

        ttk.Label(container, text="Audio Stem Splitter", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            container,
            text="Run Demucs, MDX-Net, MDX23C, and Roformer separation models.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        self._path_row(
            container,
            row=2,
            label="Input audio",
            variable=self.input_file,
            browse_command=self._choose_input,
        )

        settings = ttk.LabelFrame(container, text="Separation settings", padding=12)
        settings.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for column in range(4):
            settings.columnconfigure(column, weight=1)

        ttk.Label(settings, text="Model").grid(row=0, column=0, sticky="w")
        self.model_box = ttk.Combobox(
            settings,
            textvariable=self.model_name,
            values=MODELS,
            state="readonly",
            width=44,
        )
        self.model_box.grid(row=1, column=0, sticky="ew", padx=(0, 10))

        ttk.Label(settings, text="Shifts").grid(row=0, column=1, sticky="w")
        self.shifts_box = ttk.Spinbox(
            settings,
            from_=1,
            to=20,
            textvariable=self.shifts,
            width=8,
        )
        self.shifts_box.grid(row=1, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(settings, text="Device").grid(row=0, column=2, sticky="w")
        self.device_box = ttk.Combobox(
            settings,
            textvariable=self.device,
            values=("auto", "cpu", "cuda"),
            state="readonly",
        )
        self.device_box.grid(row=1, column=2, sticky="ew", padx=(0, 10))

        ttk.Label(settings, text="Segment (0 = default)").grid(
            row=0, column=3, sticky="w"
        )
        self.segment_box = ttk.Spinbox(
            settings,
            from_=0,
            to=999,
            textvariable=self.segment,
            width=8,
        )
        self.segment_box.grid(row=1, column=3, sticky="ew")

        ttk.Label(
            settings,
            textvariable=self.model_description,
            style="Subtitle.TLabel",
            wraplength=820,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="Run name (optional)").grid(
            row=3, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(settings, textvariable=self.run_name).grid(
            row=4, column=0, columnspan=3, sticky="ew", padx=(0, 10)
        )
        ttk.Checkbutton(settings, text="Save stems as MP3", variable=self.mp3).grid(
            row=4, column=3, sticky="w"
        )

        paths = ttk.LabelFrame(container, text="Output and evaluation", padding=12)
        paths.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        paths.columnconfigure(1, weight=1)

        self._path_row(
            paths,
            row=0,
            label="Output directory",
            variable=self.output_dir,
            browse_command=lambda: self._choose_directory(self.output_dir),
        )
        self._path_row(
            paths,
            row=1,
            label="Experiments directory",
            variable=self.experiments_dir,
            browse_command=lambda: self._choose_directory(self.experiments_dir),
        )
        self._path_row(
            paths,
            row=2,
            label="Reference stems (optional)",
            variable=self.reference_dir,
            browse_command=lambda: self._choose_directory(self.reference_dir),
        )
        self._path_row(
            paths,
            row=3,
            label="Evaluation mapping (optional)",
            variable=self.mapping_file,
            browse_command=self._choose_mapping,
        )

        controls = ttk.Frame(container)
        controls.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        controls.columnconfigure(3, weight=1)

        self.run_button = ttk.Button(
            controls,
            text="Run separation",
            command=self._start,
            style="Run.TButton",
        )
        self.run_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 8))

        ttk.Button(controls, text="Clear log", command=self._clear_log).grid(
            row=0, column=2
        )
        ttk.Label(controls, textvariable=self.status).grid(row=0, column=3, sticky="e")

        self.progress = ttk.Progressbar(container, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        log_frame = ttk.LabelFrame(container, text="Process output", padding=6)
        log_frame.grid(row=10, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            background="#101418",
            foreground="#e6edf3",
            insertbackground="#e6edf3",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    @staticmethod
    def _path_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=3
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text="Browse...", command=browse_command).grid(
            row=row, column=2, padx=(8, 0), pady=3
        )

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an audio mixture",
            initialdir=str(PROJECT_ROOT / "samples"),
            filetypes=SUPPORTED_AUDIO_TYPES,
        )
        if not selected:
            return
        self.input_file.set(selected)

        input_path = Path(selected)
        required_references = ("vocals.wav", "drums.wav", "bass.wav", "other.wav")
        if input_path.name.lower() == "mixture.wav" and all(
            (input_path.parent / name).is_file() for name in required_references
        ):
            self.reference_dir.set(str(input_path.parent))
            self._set_default_mapping()

    def _choose_directory(self, variable: tk.StringVar) -> None:
        initial_value = variable.get()
        initial_directory = initial_value if Path(initial_value).is_dir() else PROJECT_ROOT
        selected = filedialog.askdirectory(
            title="Choose a directory",
            initialdir=str(initial_directory),
        )
        if selected:
            variable.set(selected)

    def _choose_mapping(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an evaluation mapping",
            initialdir=str(PROJECT_ROOT / "configs" / "mappings"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.mapping_file.set(selected)

    def _model_changed(self, *_args) -> None:
        model = get_model_spec(self.model_name.get())
        self.model_description.set(
            f"{model.architecture}: {model.purpose}. Backend: {model.backend}."
        )
        if model.backend == "demucs":
            self.shifts_box.configure(state="normal")
            self.device_box.configure(state="readonly")
            self.segment_box.configure(state="normal")
        else:
            self.shifts_box.configure(state="disabled")
            self.device_box.configure(state="disabled")
            self.segment_box.configure(state="disabled")
        self._set_default_mapping()

    def _reference_changed(self, *_args) -> None:
        self._set_default_mapping()

    def _set_default_mapping(self) -> None:
        default_mapping = PROJECT_ROOT / "configs" / "mappings" / "demucs_6s_to_4s.json"
        if self.model_name.get() == "htdemucs_6s" and self.reference_dir.get():
            if not self.mapping_file.get():
                self.mapping_file.set(str(default_mapping))
        elif self.mapping_file.get() == str(default_mapping):
            self.mapping_file.set("")

    def _validate(self) -> bool:
        input_path = Path(self.input_file.get().strip())
        if not input_path.is_file():
            messagebox.showerror("Invalid input", "Choose an existing audio file.")
            return False
        if input_path.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            messagebox.showerror("Invalid input", "Choose a WAV, MP3, or FLAC file.")
            return False
        try:
            shifts = self.shifts.get()
            segment = self.segment.get()
        except tk.TclError:
            messagebox.showerror(
                "Invalid settings",
                "Shifts and segment must be whole numbers.",
            )
            return False

        if shifts < 1:
            messagebox.showerror("Invalid shifts", "The number of shifts must be at least 1.")
            return False
        if segment < 0:
            messagebox.showerror("Invalid segment", "Segment cannot be negative.")
            return False

        if not self.output_dir.get().strip():
            messagebox.showerror("Invalid output", "Choose an output directory.")
            return False
        if not self.experiments_dir.get().strip():
            messagebox.showerror("Invalid output", "Choose an experiments directory.")
            return False

        reference = self.reference_dir.get().strip()
        mapping = self.mapping_file.get().strip()
        if reference and not Path(reference).is_dir():
            messagebox.showerror(
                "Invalid references",
                "The reference-stem directory does not exist.",
            )
            return False
        if mapping and not reference:
            messagebox.showerror(
                "Invalid mapping",
                "Choose a reference-stem directory before selecting a mapping.",
            )
            return False
        if mapping and not Path(mapping).is_file():
            messagebox.showerror("Invalid mapping", "The mapping file does not exist.")
            return False
        return True

    def _start(self) -> None:
        if self.process is not None or not self._validate():
            return
        model = get_model_spec(self.model_name.get())
        if model.backend == "demucs":
            command = build_separation_command(
                python_executable=sys.executable,
                input_file=self.input_file.get().strip(),
                output_dir=self.output_dir.get().strip(),
                experiments_dir=self.experiments_dir.get().strip(),
                model_name=self.model_name.get(),
                device=self.device.get(),
                segment=self.segment.get(),
                shifts=self.shifts.get(),
                mp3=self.mp3.get(),
                run_name=self.run_name.get().strip(),
                reference_dir=self.reference_dir.get().strip(),
                evaluation_mapping=self.mapping_file.get().strip(),
            )
        else:
            mdx_python = (
                Path(sys.executable)
                if importlib.util.find_spec("audio_separator") is not None
                else MDX_PYTHON
            )
            if not mdx_python.is_file():
                messagebox.showerror(
                    "MDX environment not found",
                    "Create .venv-mdx and install audio-separator[cpu] or "
                    "audio-separator[gpu] before running this model.",
                )
                return
            environment_error = check_audio_separator_environment(mdx_python)
            if environment_error:
                messagebox.showerror(
                    "MDX environment is not ready",
                    f"{environment_error}\n\nRecreate .venv-mdx and install "
                    "audio-separator[cpu] or audio-separator[gpu].",
                )
                return
            command = build_audio_separator_command(
                python_executable=str(mdx_python),
                input_file=self.input_file.get().strip(),
                output_dir=self.output_dir.get().strip(),
                experiments_dir=self.experiments_dir.get().strip(),
                model_name=self.model_name.get(),
                mp3=self.mp3.get(),
                run_name=self.run_name.get().strip(),
                reference_dir=self.reference_dir.get().strip(),
                evaluation_mapping=self.mapping_file.get().strip(),
            )

        self.stop_requested = False
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)
        self.status.set("Running...")
        self._append_log(f"> {subprocess.list2cmdline(command)}\n\n")

        self.worker = threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
        )
        self.worker.start()

    def _run_process(self, command: list[str]) -> None:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            if self.stop_requested:
                self.process.terminate()
            if self.process.stdout is not None:
                for line in self.process.stdout:
                    self.messages.put(("log", line))
            return_code = self.process.wait()
            self.messages.put(("done", return_code))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def _stop(self) -> None:
        if self.worker is None:
            return
        self.stop_requested = True
        self.status.set("Stopping...")
        self._append_log("\nStopping the separation process...\n")
        if self.process is None:
            return
        try:
            self.process.terminate()
        except OSError as error:
            self._append_log(f"Could not stop process: {error}\n")

    def _poll_messages(self) -> None:
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "log":
                    self._append_log(str(payload))
                elif message_type == "done":
                    self._finish(int(payload))
                elif message_type == "error":
                    self._append_log(f"\nGUI process error: {payload}\n")
                    self._finish(-1)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _finish(self, return_code: int) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.process = None
        self.worker = None

        if self.stop_requested:
            self.status.set("Stopped")
            self._append_log("\nProcess stopped.\n")
        elif return_code == 0:
            self.status.set("Completed successfully")
            self._append_log("\nSeparation completed successfully.\n")
            messagebox.showinfo(
                "Completed",
                "Stem separation completed successfully. See the process output for paths.",
            )
        else:
            self.status.set(f"Failed (exit code {return_code})")
            self._append_log(f"\nProcess failed with exit code {return_code}.\n")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.worker is not None:
            should_close = messagebox.askyesno(
                "Separation is running",
                "Stop the current process and close the application?",
            )
            if not should_close:
                return
            self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    StemSplitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
