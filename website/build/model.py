"""Everything the site knows, read from robot_controllers itself and its results ledger.

Nothing here is written per controller: the boxes and controllers come from `robot_controllers.registry`, the
metric catalogue from `metrics.names`, parameters from each built controller's config dataclass (fields, defaults,
the comment beside each field), the rung ladder and the numbers from `results/<box-slug>/results.json`
(CONTRACT §5). A controller whose class is not importable yet still gets its page: the spec, with parameters and
the example marked as not available. Nothing here may crash the build.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")  # a build never opens a window

import robot_msgs  # noqa: E402
import robot_controllers  # noqa: E402
from robot_controllers import registry  # noqa: E402
from robot_controllers.bench.types import HostTiming  # noqa: E402
from robot_controllers.core import WANDB_ENTITY, Box, ControllerSpec  # noqa: E402
from robot_controllers.metrics.names import BY_NAME, FAMILIES, METRICS, MetricInfo  # noqa: E402

WEBSITE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = WEBSITE_DIR.parent
DEFAULT_RESULTS_DIR = REPO_DIR / "results"
CONTRACT = REPO_DIR / "docs" / "CONTRACT.md"
README = REPO_DIR / "README.md"
SCHEMA = "robot-controllers/results-1"
ROBOTMSGS_ORIGIN = "https://robotmsgs.com"
BOARD_RATE_HZ = 4000.0  # the servo-board model's rate (CONTRACT §3); shown on the host <-> bus <-> board diagram

WARNINGS: list[str] = []  # printed by the build: things the site degraded around, never fatal


def warn(message: str) -> None:
    WARNINGS.append(message)


# ─────────────────────────────────────────────────────────────────────────────────────────── formatting


def fmt(value: Any) -> str:
    """The one number format every page uses (tests format the JSON with it and compare)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    magnitude = abs(number)
    if number.is_integer() and magnitude < 1e6:
        return f"{number:.0f}"
    if magnitude >= 100:
        return f"{number:.0f}"
    if magnitude >= 10:
        return f"{number:.1f}"
    if magnitude >= 1:
        return f"{number:.2f}"
    return f"{number:.3g}"


UNIT_TEXT = {"deg": "°", "us": "µs", "count": "", "": ""}


def unit_text(unit: str) -> str:
    """Display unit: degrees as °, microseconds as µs, counts bare."""
    return UNIT_TEXT.get(unit, unit)


def with_unit(value: Any, unit: str) -> str:
    text = fmt(value)
    if value is None or text == "—":
        return text
    shown = unit_text(unit)
    if not shown:
        return text
    return f"{text}{shown}" if shown in ("°", "%") else f"{text} {shown}"


OP_TEXT = {"<=": "≤", ">=": "≥", "==": "="}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def short_name(name: str) -> str:
    """A plot label: `joint_impedance_board` -> `impedance_board`, `joint_trajectory_controller` -> `trajectory`."""
    short = re.sub(r"^joint_", "", name)
    short = re.sub(r"_controller$", "", short)
    return short or name


# ───────────────────────────────────────────────────────────────────────────────────────── robot_msgs links


def message_link(cls: type) -> tuple[str, str]:
    """(name, robotmsgs.com URL) for a robot_msgs message class: /types/<domain>/<Name>/."""
    key = next((k for k, c in robot_msgs.message_types().items() if c is cls), None)
    if key is None:
        domain = cls.__module__.split(".")[1] if cls.__module__.startswith("robot_msgs.") else "types"
        key = f"{domain}/{cls.__name__}"
    return cls.__name__, f"{ROBOTMSGS_ORIGIN}/types/{key}/"


# ───────────────────────────────────────────────────────────────────────────────────────────── parameters


@dataclass
class ParamInfo:
    name: str
    annotation: str
    default: Any  # the value this controller runs with (class default, then the spec's default_config)
    default_text: str  # how the page shows it (angles in degrees)
    unit: str
    doc: str
    set_by_spec: bool = False  # the registry spec overrides the class default


# Units live in field names (house convention); longest suffix wins.
NAME_UNITS: list[tuple[str, str]] = [
    ("_nm_s_rad", "N·m·s/rad"), ("_n_s_m", "N·s/m"), ("_nm_rad", "N·m/rad"), ("_kg_m2", "kg·m²"),
    ("_rad_s", "rad/s"), ("_m_s", "m/s"), ("_nm", "N·m"), ("_rad", "rad"), ("_deg", "deg"), ("_hz", "Hz"),
    ("_ms", "ms"), ("_us", "µs"), ("_mm", "mm"), ("_kg", "kg"), ("_s", "s"), ("_m", "m"), ("_n", "N"),
]


def unit_of(name: str) -> str:
    for suffix, unit in NAME_UNITS:
        if name.endswith(suffix):
            return unit
    return ""


def _as_floats(value: Any) -> list[float] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return [float(v) for v in value]
    return None


def value_text(value: Any, unit: str) -> str:
    """A parameter value for humans. WHY degrees: angles are shown in degrees (Zach, 2026-11-11); SI stays in the
    code, so rad values are converted here, at the display edge, with the SI value beside them."""
    numbers = _as_floats(value)
    if numbers is None:
        if isinstance(value, str):
            return repr(value)
        return "None" if value is None else repr(value)

    def joined(values: list[float]) -> str:
        return ", ".join(fmt(v) for v in values) if len(values) > 1 else fmt(values[0])

    if unit == "rad":
        return f"{joined([math.degrees(v) for v in numbers])}° ({joined(numbers)} rad)"
    if unit == "rad/s":
        return f"{joined([math.degrees(v) for v in numbers])}°/s ({joined(numbers)} rad/s)"
    if unit in ("N·m/rad", "N·m·s/rad"):
        per_deg = unit.replace("/rad", "/deg")
        return f"{joined(numbers)} {unit} ({joined([math.radians(v) for v in numbers])} {per_deg})"
    return f"{joined(numbers)} {unit}".strip()


def field_comments(cls: type) -> dict[str, str]:
    """`name: type = default  # comment` -> {name: comment}; a full-line comment block above a field counts too."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    source = inspect.cleandoc("\n" + source) if source.startswith((" ", "\t")) else source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    comments: dict[int, str] = {}
    full_line: set[int] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                row = token.start[0]
                comments[row] = token.string.lstrip("#").strip()
                if not token.line[: token.start[1]].strip():
                    full_line.add(row)
    except (tokenize.TokenError, IndentationError):
        return {}
    found: dict[str, str] = {}
    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if class_node is None:
        return {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            trailing = [comments[r] for r in range(node.lineno, (node.end_lineno or node.lineno) + 1)
                        if r in comments and r not in full_line]
            above: list[str] = []
            row = node.lineno - 1
            while row in full_line:
                above.insert(0, comments[row])
                row -= 1
            text = " ".join(trailing) or " ".join(above)
            if text:
                found[node.target.id] = text
    return found


def docstring_mentions(cls: type) -> dict[str, str]:
    """Docstring lines that start with a field name in backticks: "- `stiffness_nm_rad` K per joint ..."."""
    doc = inspect.getdoc(cls) or ""
    found: dict[str, str] = {}
    for line in doc.splitlines():
        match = re.match(r"\s*[-*]?\s*``?(\w+)``?\s*[:—–-]?\s*(.*)", line)
        if match and match.group(2):
            found.setdefault(match.group(1), match.group(2).strip())
    return found


def default_of(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return f.default_factory()  # type: ignore[misc]
        except Exception:  # noqa: BLE001  a factory we cannot call is shown as what it is
            return "<factory>"
    return dataclasses.MISSING


def params_of(spec: ControllerSpec, config_cls: type) -> list[ParamInfo]:
    comments = field_comments(config_cls)
    mentions = docstring_mentions(config_cls)
    out = []
    for f in dataclasses.fields(config_cls):
        default = default_of(f)
        set_by_spec = f.name in spec.default_config
        if set_by_spec:
            default = spec.default_config[f.name]
        unit = unit_of(f.name)
        meta = dict(f.metadata or {})
        doc = str(meta.get("doc") or comments.get(f.name) or mentions.get(f.name) or "")
        unit = str(meta.get("unit") or unit)
        shown = "required" if default is dataclasses.MISSING else value_text(default, unit)
        annotation = f.type if isinstance(f.type, str) else inspect.formatannotation(f.type)
        out.append(ParamInfo(f.name, annotation, None if default is dataclasses.MISSING else default, shown, unit, doc,
                             set_by_spec))
    return out


def params_from_record(config: dict) -> list[ParamInfo]:
    """The config the bench ran with (results ledger), when the class itself is not importable here."""
    return [ParamInfo(name, "", value, value_text(value, unit_of(name)), unit_of(name), "") for name, value in config.items()]


# ───────────────────────────────────────────────────────────────────────────────────────────── the example


EXAMPLES: dict[str, str] = {
    # One per input type, filled with the controller's name. WHY run for real at build time: a hand-written
    # example drifts from the code; this one is executed in a subprocess and its printed output is the page's.
    "JointTrajectory": '''import numpy as np
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers import controller

times = np.linspace(0.0, 1.0, 11)  # s
trajectory = JointTrajectory(t=times, joint_names=("shoulder", "elbow"),
                             positions=np.outer(times, [0.5, -0.3]))  # rad
arm = controller("{name}").build(trajectory)
state = JointState(name=("shoulder", "elbow"), position=np.zeros(2), velocity=np.zeros(2))
for _ in range(50):  # 0.5 s at 100 Hz, in this thread: on a rig, state = drive.read()
    command, defines = arm.tick(state, 0.01)  # ... then drive.send(command)
print(arm.status, command)''',
    "JointCommandSequence": '''import numpy as np
from robot_msgs.control import JointCommand, JointCommandSequence, JointCommandStep, Reached
from robot_msgs.sensor import JointState

from robot_controllers import controller

move = JointCommandStep(name="move", timeout=3.0,
                        commands=(JointCommand.to("shoulder", 0.4, speed=1.5, max_effort=12.0),
                                  JointCommand.to("elbow", -0.2, speed=1.5, max_effort=8.0)),
                        done_when=(Reached(joints=("shoulder", "elbow"), tolerance=0.0175),))
arm = controller("{name}").build(JointCommandSequence(joint_names=("shoulder", "elbow"), steps=(move,)))
state = JointState(name=("shoulder", "elbow"), position=np.zeros(2), velocity=np.zeros(2))
for _ in range(50):  # 0.5 s at 100 Hz, in this thread: on a rig, state = drive.read()
    command, defines = arm.tick(state, 0.01)  # ... then drive.send(command)
print(arm.status, command)''',
}

# Runs the example in a fresh interpreter, then reports what the built controller is: the printed output, the
# fields its last command set, whether it takes a new input (Replannable) and what it reports (SelfReport).
_RUNNER = r'''
import contextlib, io, json, math, sys
source = open(sys.argv[1]).read()
printed = io.StringIO()
namespace = {"__name__": "__example__"}
report = {"ok": False}
try:
    with contextlib.redirect_stdout(printed):
        exec(compile(source, "example.py", "exec"), namespace)
    report["ok"] = True
except BaseException as error:
    report["error"] = type(error).__name__ + ": " + str(error).splitlines()[0][:300] if str(error) else type(error).__name__
report["stdout"] = printed.getvalue()
arm = namespace.get("arm")
if arm is not None:
    report["status"] = str(getattr(arm, "status", ""))
    report["replannable"] = callable(getattr(arm, "replace_input", None))
    diagnostics = getattr(arm, "diagnostics", None)
    if callable(diagnostics):
        try:
            values = {}
            for key, value in dict(diagnostics()).items():
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    value = repr(value)
                elif isinstance(value, float) and not math.isfinite(value):
                    value = None
                values[str(key)] = value
            report["diagnostics"] = values
        except Exception as error:
            report["diagnostics_error"] = type(error).__name__ + ": " + str(error)[:200]
command = namespace.get("command")
if command is not None:
    fields = {}
    for name in ("position", "velocity", "feedforward_effort", "stiffness", "damping", "max_effort"):
        values = getattr(command, name, None)
        fields[name] = None if values is None else [float(v) if math.isfinite(float(v)) else None for v in values]
    report["command"] = fields
json.dump(report, open(sys.argv[2], "w"))
'''


@dataclass
class ExampleRun:
    code: str
    ran: bool = False
    ok: bool = False
    output: str = ""
    error: str = ""
    status: str = ""
    replannable: bool | None = None
    diagnostics: dict[str, Any] | None = None
    command: dict[str, list[float | None] | None] | None = None


def run_example(code: str, *, timeout_s: float = 90.0) -> ExampleRun:
    run = ExampleRun(code=code)
    with tempfile.TemporaryDirectory() as tmp:
        script, report_path = Path(tmp) / "example.py", Path(tmp) / "report.json"
        script.write_text(code)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
        try:
            done = subprocess.run([sys.executable, "-c", _RUNNER, str(script), str(report_path)], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            run.error = f"timed out after {timeout_s:g} s"
            return run
        if not report_path.is_file():
            run.error = (done.stderr.strip().splitlines() or ["the example runner wrote no report"])[-1]
            return run
        report = json.loads(report_path.read_text())
    run.ran = True
    run.ok = bool(report.get("ok"))
    run.output = str(report.get("stdout", "")).rstrip()
    run.error = str(report.get("error", ""))
    run.status = str(report.get("status", ""))
    run.replannable = report.get("replannable")
    run.diagnostics = report.get("diagnostics")
    run.command = report.get("command")
    return run


# ─────────────────────────────────────────────────────────────────────────────────────────────── rungs


@dataclass
class RungInfo:
    id: str
    title: str
    family: str
    order: int
    description: str
    duration_s: float | None
    params: dict[str, Any]
    timing: dict[str, Any]
    scene: dict[str, Any]
    gates: list[dict[str, Any]]  # {"metric", "op", "value", "why"}
    objective: str
    minimize: bool
    gates_text: str = ""  # contract fallback: the gate cell as written, when it could not be structured

    @property
    def anchor(self) -> str:
        return slug(self.id)

    @property
    def rate_hz(self) -> float:
        try:
            return float(self.timing.get("rate_hz") or HostTiming().rate_hz)
        except (TypeError, ValueError):
            return HostTiming().rate_hz


def rung_from_record(record: dict) -> RungInfo:
    return RungInfo(
        id=str(record.get("id", "?")), title=str(record.get("title", record.get("id", "?"))),
        family=str(record.get("family", "")), order=int(record.get("order") or 0),
        description=str(record.get("description", "")), duration_s=record.get("duration_s"),
        params=dict(record.get("params") or {}), timing=dict(record.get("timing") or {}),
        scene=dict(record.get("scene") or {}), gates=[dict(g) for g in record.get("gates") or []],
        objective=str(record.get("objective", "")), minimize=bool(record.get("minimize", True)),
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def rungs_from_bench(box: Box) -> list[RungInfo] | None:
    """The bench lane's ladder, if it imports here (it may need MuJoCo, or not be written yet)."""
    try:
        from robot_controllers.bench.rungs import ladder

        rungs = ladder(box)
    except Exception as error:  # noqa: BLE001  any failure means: use the next source
        warn(f"{box.slug}: bench ladder not importable ({type(error).__name__}: {str(error)[:120]})")
        return None
    return [rung_from_record({
        "id": r.id, "title": r.title, "family": r.family, "order": r.order, "description": r.description,
        "duration_s": r.duration_s, "params": _jsonable(dict(r.params)), "timing": _jsonable(r.timing),
        "scene": _jsonable(r.scene),
        "gates": [{"metric": g.metric, "op": g.op, "value": g.value, "why": g.why} for g in r.gates],
        "objective": r.objective, "minimize": r.minimize}) for r in rungs]


GATE_PATTERN = re.compile(r"`(\w+)\s*(<=|>=|==)\s*([-\d.]+)`")


def rungs_from_contract(box: Box) -> list[RungInfo]:
    """The ladder tables of docs/CONTRACT.md (the frozen ruler), for when nothing better is importable.

    Gates are read from the code spans of the gates cell; "same" and "rung N's gates" copy the referenced row;
    the timing gates the contract states once for the JointTrajectory ladder are added to its non-stall rungs.
    """
    try:
        text = CONTRACT.read_text()
    except OSError:
        return []
    section = text.split("### The rung ladder", 1)[-1].split("\n## ", 1)[0]
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in section.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and re.fullmatch(r"\d+", cells[0]):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    index = {"JointTrajectory": 0, "JointCommandSequence": 1}.get(box.input_name)
    if index is None or index >= len(tables):
        return []
    rows = tables[index]
    by_order: dict[int, list[dict]] = {}
    rungs = []
    previous: list[dict] = []
    for cells in rows:
        if len(cells) < 6:
            continue
        order, rung_id, title, what, gates_cell, objective_cell = cells[:6]
        rung_id = rung_id.strip("`")
        gates = [{"metric": m, "op": op, "value": float(v), "why": ""} for m, op, v in GATE_PATTERN.findall(gates_cell)]
        reference = re.search(r"rung (\d+)'s gates", gates_cell)
        if reference:
            gates = [dict(g) for g in by_order.get(int(reference.group(1)), [])]
        elif gates_cell.strip().lower() == "same":
            gates = [dict(g) for g in previous]
        by_order[int(order)] = gates
        previous = gates
        rate = 50.0 if "host50Hz" in rung_id else 500.0 if "host500Hz" in rung_id else 100.0
        all_gates = list(gates)
        if index == 0 and "Stall" not in rung_id:
            all_gates += [{"metric": "missed_cycles", "op": "==", "value": 0.0, "why": ""},
                          {"metric": "latency_p99_ms", "op": "<=", "value": round(1000.0 / rate, 4), "why": ""}]
        objective = re.search(r"`(\w+)`", objective_cell)
        rungs.append(RungInfo(
            id=rung_id, title=title, family="", order=int(order), description=re.sub(r"`", "", what),
            duration_s=None, params={}, timing={"rate_hz": rate}, scene={}, gates=all_gates,
            objective=objective.group(1) if objective else "", minimize="↑" not in objective_cell,
            gates_text=gates_cell))
    return rungs


# ─────────────────────────────────────────────────────────────────────────────────────────────── results


def load_results(results_dir: Path | None) -> dict[str, tuple[Path, dict]]:
    """{box slug: (path, document)} for every readable results-1 document under `results_dir`."""
    found: dict[str, tuple[Path, dict]] = {}
    if results_dir is None or not Path(results_dir).is_dir():
        return found
    for box in registry.BOXES:
        path = Path(results_dir) / box.slug / "results.json"
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            warn(f"{path}: unreadable ({error}); the box shows as not run")
            continue
        if document.get("schema") != SCHEMA:
            warn(f"{path}: schema {document.get('schema')!r}, expected {SCHEMA!r}; the box shows as not run")
            continue
        found[box.slug] = (path, document)
    return found


def _names_in(value: Any) -> list[str] | None:
    """A controller-name list out of whatever shape a summary field has (list of names, of dicts, or a dict)."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("front", "set", "non_dominated", "controllers", "names"):
            if key in value:
                return _names_in(value[key])
        return None
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and (item.get("controller") or item.get("name")):
                names.append(str(item.get("controller") or item.get("name")))
        return names
    return None


def non_dominated(points: dict[str, tuple[float, ...]]) -> list[str]:
    """Names whose point no other point beats or equals on every axis while differing (all axes minimized)."""
    front = []
    for name, point in points.items():
        dominated = any(
            all(o <= p for o, p in zip(other, point, strict=True)) and other != point
            for other_name, other in points.items() if other_name != name
        )
        if not dominated:
            front.append(name)
    return front


def best_of(runs: list[dict], minimize: bool, cost: dict[str, float | None] | None = None) -> str | None:
    """CONTRACT §4: the best objective among the runs that passed every gate of the rung; None if nobody did.

    Ties go to the lower compute p99, then the name (the metrics lane's summary breaks them the same way)."""
    passing = [r for r in runs if r.get("passed") and isinstance(r.get("objective"), (int, float))]
    if not passing:
        return None
    sign = 1.0 if minimize else -1.0
    cost = cost or {}

    def key(run: dict) -> tuple:
        compute = cost.get(str(run["controller"]))
        return (sign * float(run["objective"]), math.inf if compute is None else compute, str(run["controller"]))

    return str(min(passing, key=key)["controller"])


def free_space_rung(rungs: list[RungInfo]) -> RungInfo | None:
    """The tracking axis of every Pareto plot: the lowest free-space rung (R1), as the metrics lane picks it."""
    free = [r for r in rungs if r.family == "free_space"]
    if free:
        return min(free, key=lambda r: r.order)
    return next((r for r in rungs if r.id.startswith("R1")), None)


def median(values: list[float]) -> float | None:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not clean:
        return None
    middle = len(clean) // 2
    return clean[middle] if len(clean) % 2 else (clean[middle - 1] + clean[middle]) / 2


COST_METRICS = ("compute_us_p50", "compute_us_p99", "compute_us_max", "command_fields", "wire_bytes_per_tick")


# ─────────────────────────────────────────────────────────────────────────────────────── the domain model


@dataclass
class ControllerInfo:
    spec: ControllerSpec
    box_slug: str
    importable: bool = False
    import_error: str = ""
    class_summary: str = ""
    class_doc: str = ""
    params: list[ParamInfo] | None = None
    params_note: str = ""
    example: ExampleRun | None = None
    record: dict | None = None  # this controller's entry in the results document
    runs: dict[str, dict] = field(default_factory=dict)  # rung id -> RunResult dict (lowest seed)
    seeds: int = 0
    cost: dict[str, float | None] = field(default_factory=dict)  # median over rungs
    variants: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def built(self) -> bool:
        return self.spec.maturity == "built"

    @property
    def path(self) -> str:
        return f"/controllers/{self.box_slug}/{self.name}/"


@dataclass
class BoxInfo:
    box: Box
    controllers: list[ControllerInfo]
    input_link: tuple[str, str]
    output_link: tuple[str, str]
    rungs: list[RungInfo] = field(default_factory=list)
    rungs_source: str = ""  # results | bench | contract | ""
    results_path: Path | None = None
    document: dict | None = None
    best: dict[str, str | None] = field(default_factory=dict)
    pareto: dict[str, list[str]] = field(default_factory=dict)
    pareto_points: dict[str, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    recommendation: dict[str, str] | None = None
    wandb_url: str | None = None
    wandb_reports: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.box.slug

    @property
    def name(self) -> str:
        return self.box.name

    @property
    def path(self) -> str:
        return f"/boxes/{self.slug}/"

    @property
    def has_results(self) -> bool:
        return self.document is not None

    @property
    def generated(self) -> dict:
        return dict((self.document or {}).get("generated") or {})

    @property
    def fixture(self) -> bool:
        return bool(self.generated.get("fixture"))

    @property
    def built(self) -> list[ControllerInfo]:
        return [c for c in self.controllers if c.built]

    def controller(self, name: str) -> ControllerInfo | None:
        return next((c for c in self.controllers if c.name == name), None)

    def rung(self, rung_id: str) -> RungInfo | None:
        return next((r for r in self.rungs if r.id == rung_id), None)


def describe_class(info: ControllerInfo) -> None:
    spec = info.spec
    if spec.maturity != "built":
        return
    try:
        cls = spec.controller_class()
    except Exception as error:  # noqa: BLE001  the controllers lane may not have written it yet
        info.import_error = f"{type(error).__name__}: {str(error)[:200]}"
        warn(f"{spec.name}: class not importable ({info.import_error})")
        return
    info.importable = True
    doc = inspect.getdoc(cls) or ""
    if doc and not doc.startswith(f"{cls.__name__}("):
        first, _, rest = doc.partition("\n\n")
        info.class_summary = " ".join(first.split())
        info.class_doc = rest.strip()


def describe_params(info: ControllerInfo) -> None:
    spec = info.spec
    if spec.maturity != "built":
        info.params_note = "Not built yet."
        return
    if spec.config_target is None:
        info.params = []
        info.params_note = "No config: it is built from the input message alone; its parameters live in the message."
        return
    try:
        cls = spec.config_class()
        if cls is None or not dataclasses.is_dataclass(cls):
            raise TypeError(f"{spec.config_target} is not a dataclass")
        info.params = params_of(spec, cls)
        return
    except Exception as error:  # noqa: BLE001
        warn(f"{spec.name}: config not importable ({type(error).__name__}: {str(error)[:120]})")
    if info.record and isinstance(info.record.get("config"), dict) and info.record["config"]:
        info.params = params_from_record(info.record["config"])
        info.params_note = "The config class is not importable in this build: values from the results ledger."
    else:
        info.params = None
        info.params_note = "Parameters not available: the config class is not importable yet."


def attach_results(box_info: BoxInfo, path: Path, document: dict) -> None:
    box_info.results_path = path
    box_info.document = document
    records = {str(c.get("name")): c for c in document.get("controllers") or [] if isinstance(c, dict)}
    rows = [r for r in document.get("results") or [] if isinstance(r, dict)]
    known = {c.name for c in box_info.controllers}
    for name in sorted({str(r.get("controller")) for r in rows} - known):
        warn(f"{box_info.slug}: results for {name!r}, which the registry does not list; not shown")
    for info in box_info.controllers:
        info.record = records.get(info.name)
        mine = [r for r in rows if r.get("controller") == info.name]
        info.seeds = len({r.get("seed") for r in mine})
        for row in sorted(mine, key=lambda r: (r.get("seed") is None, r.get("seed") or 0), reverse=True):
            info.runs[str(row.get("rung"))] = row  # WHY reverse then overwrite: the lowest seed ends up shown
        for metric in COST_METRICS:
            info.cost[metric] = median([(r.get("metrics") or {}).get(metric) for r in info.runs.values()])

    for rung in box_info.rungs:
        runs = [c.runs[rung.id] for c in box_info.controllers if rung.id in c.runs]
        box_info.best[rung.id] = best_of(runs, rung.minimize, {c.name: c.cost.get("compute_us_p99") for c in box_info.controllers})

    summary = document.get("summary") or {}
    per_rung = summary.get("per_rung") or {}
    for rung_id, mine in box_info.best.items():
        theirs = (per_rung.get(rung_id) or {}).get("best") if isinstance(per_rung.get(rung_id), dict) else None
        theirs_names = _names_in(theirs) if theirs is not None else None
        if theirs_names is not None and theirs_names[:1] != ([mine] if mine else []):
            warn(f"{box_info.slug} {rung_id}: summary best {theirs_names} differs from the site's {mine!r}")

    # Pareto (CONTRACT §4): per belt rung, (R1 tracking p95, that rung's largest contact, compute p99)
    anchor = free_space_rung(box_info.rungs)
    summary_pareto = summary.get("pareto") or {}
    for rung in box_info.rungs:
        if rung.objective != "contact_max_n" or anchor is None:
            continue
        points = {}
        for info in box_info.built:
            track = ((info.runs.get(anchor.id) or {}).get("metrics") or {}).get("tracking_p95_deg")
            contact = ((info.runs.get(rung.id) or {}).get("metrics") or {}).get("contact_max_n")
            compute = info.cost.get("compute_us_p99")
            if all(isinstance(v, (int, float)) for v in (track, contact, compute)):
                points[info.name] = (float(track), float(contact), float(compute))
        if not points:
            continue
        box_info.pareto_points[rung.id] = points
        from_summary = _names_in(summary_pareto.get(rung.id)) if isinstance(summary_pareto, dict) else None
        if from_summary is None:
            # WHY among the rung's passers when there are any (the metrics lane's rule): a controller that failed
            # the belt gates, e.g. never touched the belt (0 N), would sit on the front for not doing the task.
            passers = {n: p for n, p in points.items() if (box_info.controller(n).runs.get(rung.id) or {}).get("passed")}
            from_summary = non_dominated(passers or points)
        box_info.pareto[rung.id] = from_summary

    recommendation = summary.get("recommendation")
    if isinstance(recommendation, dict) and (recommendation.get("controller") or recommendation.get("name")):
        box_info.recommendation = {"controller": str(recommendation.get("controller") or recommendation.get("name")),
                                   "why": str(recommendation.get("why") or ""), "source": "summary"}
    elif isinstance(recommendation, str):
        box_info.recommendation = {"controller": recommendation, "why": "", "source": "summary"}
    else:
        box_info.recommendation = recommend(box_info)

    wandb = document.get("wandb") or {}
    url = wandb.get("project_url")
    box_info.wandb_url = rewrite_wandb(url) if isinstance(url, str) and url.startswith("https://") else None
    reports = wandb.get("reports") or {}
    if isinstance(reports, dict):
        box_info.wandb_reports = {str(k): rewrite_wandb(v) for k, v in reports.items()
                                  if isinstance(v, str) and v.startswith("https://")}


def rewrite_wandb(url: str) -> str:
    """CONTRACT §6: W&B links point at wandb.ai, never the forge host the API sometimes returns."""
    return re.sub(r"^https://[^/]*forge\.coreweave\.com", "https://wandb.ai", url)


def recommend(box_info: BoxInfo) -> dict[str, str] | None:
    """CONTRACT §4's rule, when the ledger has no summary: most rungs passed, then most gates, then lowest compute."""
    scored = []
    for info in box_info.built:
        runs = list(info.runs.values())
        if not runs:
            continue
        rungs_passed = sum(bool(r.get("passed")) for r in runs)
        gates = [g for r in runs for g in r.get("gates") or []]
        gates_passed = sum(bool(g.get("passed")) for g in gates)
        compute = info.cost.get("compute_us_p99")
        scored.append((-rungs_passed, -gates_passed, compute if compute is not None else math.inf, info.name,
                       f"passes {rungs_passed}/{len(runs)} rungs and {gates_passed}/{len(gates)} gates at "
                       f"{fmt(compute)} µs p99 per tick"))
    if not scored:
        return None
    best = sorted(scored)[0]
    return {"controller": best[3], "why": best[4], "source": "site"}


def collect(results_dir: Path | None = DEFAULT_RESULTS_DIR, *, run_examples: bool = True) -> list[BoxInfo]:
    WARNINGS.clear()
    documents = load_results(results_dir)
    boxes = []
    for box in registry.BOXES:
        infos = [ControllerInfo(spec=s, box_slug=box.slug) for s in registry.in_box(box)]
        for info in infos:
            info.variants = [s.name for s in registry.in_box(box) if s.variant_of == info.name]
        box_info = BoxInfo(box=box, controllers=infos, input_link=message_link(box.input_type),
                           output_link=message_link(box.output_type))
        if box.slug in documents:
            path, document = documents[box.slug]
            box_info.rungs = sorted((rung_from_record(r) for r in document.get("rungs") or [] if isinstance(r, dict)),
                                    key=lambda r: r.order)
            box_info.rungs_source = "results"
        elif any(i.built for i in infos):
            bench = rungs_from_bench(box)
            if bench:
                box_info.rungs, box_info.rungs_source = sorted(bench, key=lambda r: r.order), "bench"
            else:
                contract = rungs_from_contract(box)
                if contract:
                    box_info.rungs, box_info.rungs_source = contract, "contract"
        if box.slug in documents:
            attach_results(box_info, *documents[box.slug])
        for info in infos:
            describe_class(info)
            describe_params(info)
            template = EXAMPLES.get(box.input_name)
            if info.built and template:
                code = template.replace("{name}", info.name)
                info.example = run_example(code) if (run_examples and info.importable) else ExampleRun(code=code)
                if not info.importable:
                    info.example.error = info.import_error or "the class is not importable"
                elif not run_examples:
                    info.example.error = "examples were skipped in this build (--no-examples)"
        boxes.append(box_info)
    return boxes


# ─────────────────────────────────────────────────────────────────────────────────────────── home copy


@dataclass(frozen=True)
class Intro:
    """The home page's words, read from the package (WHY: hand-written copy on robotmsgs.com went stale)."""

    tagline: str
    eyebrow: str
    lead: str
    pillars: list[tuple[str, str]]
    dependencies: list[str]
    version: str


def intro() -> Intro:
    from importlib.metadata import PackageNotFoundError, metadata, requires

    try:
        meta = metadata("robot-controllers")
        summary, version = meta["Summary"], meta["Version"]
        python = (meta.get("Requires-Python") or "").replace(">=", "≥ ")
        license_name = meta.get("License-Expression") or meta.get("License") or ""
        dependencies = [r.split(";")[0].strip() for r in requires("robot-controllers") or [] if "extra ==" not in r]
    except PackageNotFoundError:
        summary, version, python, license_name, dependencies = "", robot_controllers.__version__, "", "", []
    blocks = README.read_text().split("\n\n") if README.is_file() else ["", "", ""]
    lead = " ".join(blocks[1].split()) if len(blocks) > 1 else ""
    pillars = []
    if len(blocks) > 2:
        for item in re.split(r"\n(?=\d+\.\s|- )", "\n" + blocks[2].strip())[1:]:
            text = " ".join(re.sub(r"^(\d+\.|-)\s+", "", item.strip()).split())
            match = re.match(r"\*\*(.+?)\*\*\s*(.*)", text)
            if match:
                pillars.append((match.group(1).rstrip("."), match.group(2)))
    eyebrow = " · ".join(p for p in (f"Python {python}".strip() if python else "", license_name, f"v{version}") if p)
    # WHY the part before the colon: the pyproject summary is a full sentence; the hero needs a name-sized line and
    # the README lead under it already says the rest.
    tagline = (summary or lead).split(":")[0].strip()
    return Intro(tagline=tagline, eyebrow=eyebrow, lead=lead, pillars=pillars, dependencies=dependencies,
                 version=version)


def metrics_by_family() -> list[tuple[str, list[MetricInfo]]]:
    return [(family, [m for m in METRICS if m.family == family]) for family in FAMILIES]


def metric(name: str) -> MetricInfo | None:
    return BY_NAME.get(name)


def wandb_project_url(box: Box) -> str:
    """Where the box's project would live on W&B; shown as text only until the ledger says it was logged."""
    return f"https://wandb.ai/{WANDB_ENTITY}/{box.wandb_project}"


def source_stamp() -> str:
    """'robot_controllers 0.1.0 @ 101230a' for the footer; the commit part is best effort."""
    version = f"robot_controllers {robot_controllers.__version__}"
    try:
        sha = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return version
    return f"{version} @ {sha}"
