"""The joint_trajectory_controller law, compiled: a second implementation of the same black box (CONTRACT
amendment 2).

`JointTrajectoryControllerCpp(trajectory, JointTrajectoryConfig)` is the Python `JointTrajectoryController`
with its law in C++ (`cpp/include/robot_controllers/joint_trajectory_controller.hpp`, header-only, no
allocation in a tick). The same header runs inside the bam_ws moteus controller node's real-time thread; this
adapter puts it on the ladder so both implementations are ranked on the same rungs, and
tests/test_cpp_parity.py pins it to the Python law (identical ServoCommands, 1e-12; bit-exact today).

WHY ctypes and a compile on first use: no new Python dependency and no build backend change. The C++ sources
ship inside the package; the first build writes `librobot_controllers_jtc-<hash>.so` to the cache
(`$ROBOT_CONTROLLERS_CACHE`, else `$XDG_CACHE_HOME/robot_controllers`, else `~/.cache/robot_controllers`),
keyed by the sources and flags so an edit rebuilds. `$ROBOT_CONTROLLERS_JTC_LIB` points at a library built
elsewhere (the CMake build in `cpp/CMakeLists.txt`).

Python-side validation runs first (the Python sampler and `per_joint`), so every refusal raises the very same
exception and changes nothing. One difference, deliberate: a state missing one of the trajectory's joints
raises its KeyError before the clock advances (the Python law advances first).
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
import weakref
from pathlib import Path

import numpy as np
from robot_msgs.control import ServoCommand
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.controllers._sampling import JointSelector, TrajectorySampler, check_dt, per_joint
from robot_controllers.controllers.joint_trajectory import JointTrajectoryConfig
from robot_controllers.core import Status

CPP_DIR = Path(__file__).resolve().parent / "cpp"
HEADER = CPP_DIR / "include" / "robot_controllers" / "joint_trajectory_controller.hpp"
SHIM = CPP_DIR / "jtc_capi.cpp"
ABI_VERSION = 1
# WHY these flags: -ffp-contract=off keeps (1 - w) * a + w * b free of fused multiply-adds, so the samples
# round exactly as numpy's; -O2 without -ffast-math keeps IEEE semantics (NaN measurements must stay NaN).
COMPILE_FLAGS = ("-std=c++17", "-O2", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math")
_STATUS: dict[int, Status] = {0: "running", 1: "done", 2: "aborted"}
_LOAD_ERRORS = {
    1: "empty trajectory",
    2: "no joints",
    3: "more joints than the C++ law holds",
    4: "missing array",
}

_library: ctypes.CDLL | None = None
_library_lock = threading.Lock()


def cache_dir() -> Path:
    if os.environ.get("ROBOT_CONTROLLERS_CACHE"):
        return Path(os.environ["ROBOT_CONTROLLERS_CACHE"])
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "robot_controllers"


def _compiler() -> str:
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and shutil.which(name):
            return shutil.which(name)  # type: ignore[return-value]
    raise RuntimeError(
        "joint_trajectory_controller_cpp needs a C++17 compiler (c++, g++ or clang++) to build its law on "
        "first use, or ROBOT_CONTROLLERS_JTC_LIB pointing at a built librobot_controllers_jtc.so"
    )


def build_library() -> Path:
    """The shared library for the current sources, compiled into the cache when it is not there yet."""
    compiler = _compiler()
    digest = hashlib.sha1()
    for part in (HEADER.read_bytes(), SHIM.read_bytes(), " ".join(COMPILE_FLAGS).encode(), compiler.encode()):
        digest.update(part)
    target = cache_dir() / f"librobot_controllers_jtc-{digest.hexdigest()[:16]}.so"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # WHY a temporary name and a rename: two processes compiling at once each finish their own file and the
    # second rename wins; a reader never loads half a library.
    handle, temporary = tempfile.mkstemp(prefix=target.stem + ".", suffix=".tmp.so", dir=target.parent)
    os.close(handle)
    try:
        command = [compiler, *COMPILE_FLAGS, f"-I{CPP_DIR / 'include'}", str(SHIM), "-o", temporary]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"building {SHIM.name} failed ({' '.join(command)}):\n{result.stderr}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def load_library() -> ctypes.CDLL:
    """The C ABI, loaded once per process (`$ROBOT_CONTROLLERS_JTC_LIB`, else built into the cache)."""
    global _library
    with _library_lock:
        if _library is not None:
            return _library
        path = os.environ.get("ROBOT_CONTROLLERS_JTC_LIB") or str(build_library())
        library = ctypes.CDLL(path)
        if library.rc_jtc_abi_version() != ABI_VERSION:
            version = library.rc_jtc_abi_version()
            raise RuntimeError(f"{path} speaks ABI {version}, this adapter {ABI_VERSION}")
        pointer = ctypes.c_void_p
        library.rc_jtc_create.restype = pointer
        library.rc_jtc_destroy.argtypes = [pointer]
        library.rc_jtc_destroy.restype = None
        sizes = [ctypes.c_size_t, ctypes.c_size_t]
        library.rc_jtc_load.argtypes = [pointer, ctypes.c_int, *sizes] + [pointer] * 6
        library.rc_jtc_load.restype = ctypes.c_int
        library.rc_jtc_tick.argtypes = [pointer, pointer, ctypes.c_double, pointer, pointer, pointer]
        library.rc_jtc_tick.restype = ctypes.c_int
        for name, restype in (
            ("rc_jtc_status", ctypes.c_int),
            ("rc_jtc_clock", ctypes.c_double),
            ("rc_jtc_end_time", ctypes.c_double),
            ("rc_jtc_active_input", ctypes.c_int),
            ("rc_jtc_worst_joint", ctypes.c_int),
            ("rc_jtc_worst_error", ctypes.c_double),
            ("rc_jtc_law_cpu_ns", ctypes.c_longlong),
        ):
            function = getattr(library, name)
            function.argtypes = [pointer]
            function.restype = restype
        library.rc_jtc_max_joints.restype = ctypes.c_int
        _library = library
        return library


def _contiguous(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.ascontiguousarray(values, dtype=np.float64)


def _address(values: np.ndarray | None) -> int | None:
    return None if values is None else values.ctypes.data


class JointTrajectoryControllerCpp:
    """`JointTrajectoryController` with the law in C++ (same config, same commands, same diagnostics plus
    `law_cpu_ns`: the thread CPU time of the last tick's law inside C++, without the ctypes and ServoCommand
    overhead the bench's compute_s includes)."""

    _mode = "time_driven"
    implementation = "cpp"

    def __init__(self, trajectory: JointTrajectory, config: JointTrajectoryConfig | None = None) -> None:
        self.config = config if config is not None else JointTrajectoryConfig()
        self._library = load_library()
        self._handle = self._library.rc_jtc_create()
        if not self._handle:
            raise MemoryError("rc_jtc_create returned null")
        weakref.finalize(self, self._library.rc_jtc_destroy, self._handle)
        self.status: Status = "running"
        self.message = ""
        self.active_input = 0
        self._load(trajectory, replace=False)

    def _load(self, trajectory: JointTrajectory, *, replace: bool) -> None:
        # Everything that can fail is checked before anything is assigned: a refused input changes nothing.
        sampler = TrajectorySampler(trajectory)
        names = sampler.joint_names
        max_effort = per_joint(self.config.max_effort_nm, names, "max_effort_nm")
        tolerance = per_joint(self.config.path_tolerance_rad, names, "path_tolerance_rad")
        if len(names) > self._library.rc_jtc_max_joints():
            limit = self._library.rc_jtc_max_joints()
            raise ValueError(f"{len(names)} joints; the C++ law holds at most {limit}")
        template = ServoCommand(
            name=names, position=np.zeros(len(names)), velocity=np.zeros(len(names)), max_effort=max_effort
        )
        # The C++ law reads these in place: they live as long as this input is loaded (self._arrays).
        arrays = (
            _contiguous(trajectory.t),
            _contiguous(trajectory.positions),
            _contiguous(trajectory.velocities),
            _contiguous(trajectory.effort),
            _contiguous(tolerance),
            _contiguous(max_effort),
        )
        code = self._library.rc_jtc_load(
            self._handle, int(replace), len(trajectory), len(names), *(_address(a) for a in arrays)
        )
        if code != 0:
            raise ValueError(f"the C++ law refused the trajectory: {_LOAD_ERRORS.get(code, code)}")
        self.trajectory = trajectory
        self._arrays = arrays
        self._names = template.name
        self._selector = JointSelector(names)
        self._tolerance = tolerance
        self._max_effort = template.max_effort
        self._rest = template.velocity
        self._end_time = sampler.end_time
        self._hold: ServoCommand | None = None
        self.status = "running"
        self.message = ""

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        dt = check_dt(dt)
        if self._tolerance is not None and self._hold is None:
            measured = np.ascontiguousarray(self._selector.positions(state), dtype=np.float64)
        else:
            self._selector.check(state)
            measured = None
        count = len(self._names)
        position, velocity, effort = np.empty(count), np.empty(count), np.empty(count)
        code = self._library.rc_jtc_tick(
            self._handle,
            _address(measured),
            dt,
            position.ctypes.data,
            velocity.ctypes.data,
            effort.ctypes.data,
        )
        if code < 0:
            raise ValueError(f"dt must be finite and >= 0 s, got {dt!r}")
        if self._hold is not None:
            return self._hold, {}
        status = _STATUS[code]
        if status == "aborted":
            return self._abort(position), {}
        if status == "done" and self.status == "running":
            self.status = "done"
            self.message = f"trajectory ended at {self._end_time:.3f} s; holding its last sample"
        return (
            ServoCommand(
                name=self._names,
                position=position,
                velocity=velocity,
                feedforward_effort=effort,
                max_effort=self._max_effort,
            ),
            {},
        )

    def _abort(self, measured: np.ndarray) -> ServoCommand:
        worst = self._library.rc_jtc_worst_joint(self._handle)
        error = self._library.rc_jtc_worst_error(self._handle)
        self.status = "aborted"
        self.message = (
            f"{self._names[worst]} was {math.degrees(error):.2f} deg off the path "
            f"(tolerance {math.degrees(self._tolerance[worst]):.2f} deg); holding where it was measured"
        )
        self._hold = ServoCommand(
            name=self._names,
            position=measured,
            velocity=self._rest,
            feedforward_effort=self._rest,
            max_effort=self._max_effort,
        )
        return self._hold

    def replace_input(self, message: JointTrajectory) -> None:
        """Run `message` from now: its t = 0 is the next tick. Accepted in any status."""
        self._load(message, replace=True)
        self.active_input += 1

    @property
    def clock_s(self) -> float:
        return float(self._library.rc_jtc_clock(self._handle))

    def diagnostics(self) -> dict[str, float | str]:
        return {
            "mode": self._mode,
            "status": self.status,
            "clock_s": self.clock_s,
            "active_trajectory": self.active_input,
            "trajectory_end_s": self._end_time,
            "message": self.message,
            "implementation": self.implementation,
            "law_cpu_ns": int(self._library.rc_jtc_law_cpu_ns(self._handle)),
        }
