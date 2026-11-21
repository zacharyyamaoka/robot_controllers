"""Hero video for the study report: the UR5e on U2 (fast moves) at the pick cycle's 50 Hz, three layouts side by side.

    MUJOCO_GL=egl env -u VIRTUAL_ENV OMP_NUM_THREADS=1 uv run --frozen --directory /home/bam/robot_controllers \
        --extra test --extra bench python docs/ur5e_hero_video.py \
        --out /home/bam/robot_controllers/reports/media/controller-layouts-2026-11-20

Every frame comes from a real bench episode (`run_episode` on `bench.layouts.layout_spec(..., arm="ur5e")`, CONTRACT
amendment 4): the solid arm takes the plant's logged joint angles, the faint arm is the reference the controller was
asked to follow, and the trace is the tool's distance from the reference tool position (the `tool_*_mm` metrics' own
error). WHY Menagerie's meshed UR5e for the picture while the physics runs on the vendored bare model: the two share
kinematics and joint names (the bare model is Menagerie's links with the meshes taken out), so the meshes only change
what you see. The Menagerie checkout is a render-time asset (~/.cache/mujoco_menagerie), never a runtime dependency.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from robot_controllers.bench.layouts import layout_spec  # noqa: E402
from robot_controllers.bench.loop import run_episode  # noqa: E402
from robot_controllers.bench.rungs import rate_variant  # noqa: E402
from robot_controllers.bench.ur5e import UR5E_JOINTS, ur5e_rungs  # noqa: E402
from robot_controllers.metrics.episode import episode_metrics  # noqa: E402

MENAGERIE = Path.home() / ".cache" / "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"
COLOURS = ("#f97316", "#60a5fa", "#22c55e")
PANEL_W, PANEL_H, PLOT_H = 480, 420, 260
FPS = 30
SLOWDOWN = 2.0  # WHY: U2's moves are 0.45 s each; at real speed a move is 13 frames
GHOST_ALPHA = 0.30
LAYOUTS = (
    ("as built: no lookahead", 0.0, "tangent"),
    ("+ 5.5 ms lookahead", 0.0055, "tangent"),
    ("+ lookahead + secant velocity", 0.0055, "secant_mid_ff"),
)


def render_model() -> tuple[mujoco.MjModel, np.ndarray]:
    spec = mujoco.MjSpec.from_file(str(MENAGERIE))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    model = spec.compile()
    qpos = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in UR5E_JOINTS])
    return model, qpos


def at(times: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(times, t, values[:, j]) for j in range(values.shape[1])], axis=1)


def panel_frames(model, qpos_index, log, times):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (-0.12, 0.05, 0.32)
    camera.distance = 2.15
    camera.azimuth = 125.0
    camera.elevation = -18.0
    q = at(times, np.asarray(log.t), np.asarray(log.q))
    ref = at(times, np.asarray(log.sent_t), np.asarray(log.ref_position))
    robot = model.geom_bodyid != 0

    def shot(joints):
        data.qpos[qpos_index] = joints
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        renderer.disable_segmentation_rendering()
        colour = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        mask = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[:, :, 0] >= 0)
        mask &= robot[np.clip(seg[:, :, 0], 0, model.ngeom - 1)]
        return colour, mask

    frames = []
    for k in range(len(times)):
        actual, actual_mask = shot(q[k])
        ghost, ghost_mask = shot(ref[k])
        out = actual.astype(float)
        blend = ghost_mask & ~actual_mask  # the ghost shows where the real arm is not (it sits behind)
        out[blend] = (1 - GHOST_ALPHA) * out[blend] + GHOST_ALPHA * 255.0
        frames.append(out.astype(np.uint8))
    renderer.close()
    return frames


def tool_error_mm(log) -> tuple[np.ndarray, np.ndarray]:
    sent = np.asarray(log.sent_t)
    tool = at(sent, np.asarray(log.t), np.asarray(log.tool_pos))
    return sent, 1000.0 * np.linalg.norm(np.asarray(log.ref_tool_pos) - tool, axis=1)


def error_plot(series, times, t_now, width):
    fig, ax = plt.subplots(figsize=(width / 100, PLOT_H / 100), dpi=100)
    fig.patch.set_facecolor("#0b0b0c")
    ax.set_facecolor("#0b0b0c")
    top = 0.0
    for (label, t, err), colour in zip(series, COLOURS, strict=False):
        ax.plot(t, err, color=colour, lw=1.6, label=label)
        top = max(top, float(np.percentile(err, 99.5)))
    ax.axvline(t_now, color="#e4e4e7", lw=1.0)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, max(top * 1.15, 1.0))
    ax.set_xlabel("time (s)", color="#a1a1aa")
    ax.set_ylabel("tool error (mm)", color="#a1a1aa")
    ax.tick_params(colors="#a1a1aa")
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.legend(loc="upper right", fontsize=9, facecolor="#18181b", edgecolor="#3f3f46", labelcolor="#e4e4e7")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--slug", default="hero-ur5e")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fast = {r.id: r for r in ur5e_rungs()}["U2.FastFreeSpace"]
    rung = rate_variant(fast, args.rate_hz)
    model, qpos_index = render_model()
    runs = []
    for label, lookahead, interpolation in LAYOUTS:
        spec = layout_spec("trajectory", "trajectory", "board_gains", "board", lookahead_s=lookahead,
                           interpolation=interpolation, arm="ur5e")
        log = run_episode(spec, rung, seed=0)
        runs.append((label, spec.name, log, episode_metrics(log, rung)))
    duration = min(float(log.t[-1]) for _, _, log, _ in runs)
    times = np.arange(0.0, duration, 1.0 / (FPS * SLOWDOWN))
    panels = [panel_frames(model, qpos_index, log, times) for _, _, log, _ in runs]
    series = [(label, *tool_error_mm(log)) for label, _, log, _ in runs]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    width = PANEL_W * len(runs)
    with tempfile.TemporaryDirectory() as tmp:
        for index, t in enumerate(times):
            row = np.concatenate([frames[index] for frames in panels], axis=1)
            plot = np.asarray(Image.fromarray(error_plot(series, times, t, width)).resize((row.shape[1], PLOT_H)))
            frame = Image.fromarray(np.concatenate([row, plot], axis=0))
            draw = ImageDraw.Draw(frame)
            for column, ((label, _name, _log, metrics), colour) in enumerate(zip(runs, COLOURS, strict=False)):
                x = column * PANEL_W + 12
                draw.text((x, 10), label, fill=colour, font=font)
                draw.text((x, 32), f"p95 {metrics['tracking_p95_deg']:.2f} deg · tool p95 {metrics['tool_p95_mm']:.1f} mm",
                          fill="#e4e4e7", font=small)
            draw.text((12, PANEL_H - 24), f"UR5e · U2 at {args.rate_hz:g} Hz · t = {t:.2f} s · {SLOWDOWN:g}x slow motion · "
                      "faint arm = the reference", fill="#d4d4d8", font=small)
            frame.save(Path(tmp) / f"f{index:05d}.png")
        mp4, gif, poster = (args.out / f"{args.slug}.{ext}" for ext in ("mp4", "gif", "jpg"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=10,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                        str(gif)], check=True)
        worst = int(np.argmax(series[0][2]))
        poster_index = int(np.clip(np.searchsorted(times, series[0][1][worst]), 0, len(times) - 1))
        Image.open(Path(tmp) / f"f{poster_index:05d}.png").convert("RGB").save(poster, quality=85)
    for label, name, _log, metrics in runs:
        print(f"{label:34s} {name:80s} p95 {metrics['tracking_p95_deg']:.3f} deg  tool {metrics['tool_p95_mm']:.2f} mm")
    for path in (mp4, gif, poster):
        print(path, f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
