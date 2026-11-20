"""Hero video for the layouts report: one rung, several layouts of the same FF and FB models, side by side.

    MUJOCO_GL=egl env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test --extra bench \
        python docs/layouts_hero_video.py --out /home/bam/robot_controllers/reports/media/controller-layouts-2026-11-20

Every frame comes from a real bench episode (`run_episode` on `bench.layouts.layout_spec`): the arm's joint angles are
the plant's logged samples, the faint arm is the reference the controller was asked to follow, and the trace is the
worst joint's tracking error. Rendering adds a light and a floor; it never changes the numbers.
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

from robot_controllers.bench.arm import ARM, build_mjcf  # noqa: E402
from robot_controllers.bench.layouts import layout_spec  # noqa: E402
from robot_controllers.bench.loop import run_episode  # noqa: E402
from robot_controllers.bench.rungs import rung as rung_by_id  # noqa: E402
from robot_controllers.metrics.episode import episode_metrics  # noqa: E402

COLOURS = ("#f97316", "#22c55e", "#60a5fa", "#e879f9")
PANEL_W, PANEL_H, PLOT_H = 480, 400, 280
FPS = 30
SLOWDOWN = 2.0  # WHY: R2's moves are 0.3 s each; at real speed a whole move is nine frames


def render_model() -> tuple[mujoco.MjModel, dict[str, int]]:
    """The free-space MJCF plus a light, a floor and a ghost copy of the two links (visual only) for the reference.

    Returns the model and each joint's qpos index by name. WHY by name: the ghost sits first in the worldbody, so
    its joints take qpos 0 and 1 and the arm's move to 2 and 3.
    """
    xml = build_mjcf(belt=False, arm=ARM)
    ghost = (
        f'<body name="ghost_base" pos="0 0.02 {ARM.base_z_m}">'
        '<joint name="ghost_shoulder" type="hinge" axis="0 -1 0"/>'
        f'<geom type="capsule" fromto="0 0 0 {ARM.link1_length_m} 0 0" size="0.012" rgba="1 1 1 0.18" contype="0" conaffinity="0"/>'
        f'<body pos="{ARM.link1_length_m} 0 0"><joint name="ghost_elbow" type="hinge" axis="0 -1 0"/>'
        f'<geom type="capsule" fromto="0 0 0 {ARM.link2_length_m} 0 0" size="0.010" rgba="1 1 1 0.18" contype="0" conaffinity="0"/>'
        "</body></body>"
    )
    xml = xml.replace(
        "<worldbody>",
        '<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="2048"/></visual>'
        "<worldbody>"
        '<light pos="0.3 -1.2 1.4" dir="0 0.6 -0.8" diffuse="0.9 0.9 0.9" castshadow="true"/>'
        '<geom type="plane" size="2 2 0.01" pos="0 0 0" rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>'
        + ghost,
        1,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    names = ("shoulder", "elbow", "ghost_shoulder", "ghost_elbow")
    return model, {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names}


def reference_at(log, t: np.ndarray) -> np.ndarray:
    """The intended reference (the rung's schedule, logged at each send time), linear between sends."""
    sent = np.asarray(log.sent_t)
    ref = np.asarray(log.ref_position)
    return np.stack([np.interp(t, sent, ref[:, j]) for j in range(ref.shape[1])], axis=1)


def panel_frames(model, qpos, log, times):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.27, 0.0, 0.36)
    camera.distance = 0.85
    camera.azimuth = 90.0
    camera.elevation = -6.0
    ref = reference_at(log, times)
    frames = []
    for k, t in enumerate(times):
        data.qpos[qpos["shoulder"]] = np.interp(t, log.t, log.q[:, 0])
        data.qpos[qpos["elbow"]] = np.interp(t, log.t, log.q[:, 1])
        data.qpos[qpos["ghost_shoulder"]] = ref[k, 0]
        data.qpos[qpos["ghost_elbow"]] = ref[k, 1]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def error_deg(log) -> tuple[np.ndarray, np.ndarray]:
    """The tracking metric's own error: reference at each send time minus the arm's position at that time."""
    sent = np.asarray(log.sent_t)
    q = np.stack([np.interp(sent, log.t, log.q[:, j]) for j in range(log.q.shape[1])], axis=1)
    err = np.degrees(np.abs(np.asarray(log.ref_position) - q))
    return sent, err.max(axis=1)


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
    ax.set_ylim(0, max(top * 1.15, 0.5))
    ax.set_xlabel("time (s)", color="#a1a1aa")
    ax.set_ylabel("tracking error, worst joint (deg)", color="#a1a1aa")
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
    parser.add_argument("--rung", default="R2.FastFreeSpace")
    parser.add_argument("--slug", default="hero-layouts")
    parser.add_argument(
        "--layout",
        action="append",
        metavar="LABEL=FF_MODEL,FF_AT,FB_MODEL,FB_AT[,LOOKAHEAD_S]",
        help="repeat 2-4 times; e.g. 'simple JTC, no FF=none,none,board_gains,board'",
    )
    args = parser.parse_args()
    layouts = args.layout or [
        "JTC, no feed-forward=none,none,board_gains,board",
        "JTC as built (model torque)=trajectory,trajectory,board_gains,board",
        "JTC + 5.5 ms lookahead=trajectory,trajectory,board_gains,board,0.0055",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    rung = rung_by_id(args.rung)
    model, qpos = render_model()
    runs = []
    for item in layouts:
        label, spec_args = item.rsplit("=", 1)
        parts = [part.strip() for part in spec_args.split(",")]
        lookahead = float(parts[4]) if len(parts) > 4 else 0.0
        spec = layout_spec(*parts[:4], lookahead_s=lookahead)
        log = run_episode(spec, rung, seed=0)
        runs.append((label.strip(), spec.name, log, episode_metrics(log, rung)))
    duration = min(float(log.t[-1]) for _, _, log, _ in runs)
    times = np.arange(0.0, duration, 1.0 / (FPS * SLOWDOWN))
    panels = [panel_frames(model, qpos, log, times) for _, _, log, _ in runs]
    series = [(label, *error_deg(log)) for label, _, log, _ in runs]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()

    width = PANEL_W * len(runs)
    with tempfile.TemporaryDirectory() as tmp:
        for index, t in enumerate(times):
            row = np.concatenate([frames[index] for frames in panels], axis=1)
            plot = error_plot(series, times, t, width)
            plot = np.asarray(Image.fromarray(plot).resize((row.shape[1], PLOT_H)))
            frame = Image.fromarray(np.concatenate([row, plot], axis=0))
            draw = ImageDraw.Draw(frame)
            for column, ((label, _name, _log, metrics), colour) in enumerate(zip(runs, COLOURS, strict=False)):
                x = column * PANEL_W + 12
                draw.text((x, 10), label, fill=colour, font=font)
                draw.text((x, 32), f"p95 error {metrics['tracking_p95_deg']:.2f} deg · feedback share "
                          f"{metrics['fb_share_pct']:.0f} %", fill="#e4e4e7", font=small)
                draw.text((x, 50), f"host CPU {metrics['host_duty_pct']:.2f} % of a core", fill="#a1a1aa", font=small)
            draw.text((12, PANEL_H - 24), f"{args.rung} · t = {t:.2f} s · {SLOWDOWN:g}x slow motion · faint arm = "
                      "the reference", fill="#a1a1aa", font=small)
            frame.save(Path(tmp) / f"f{index:05d}.png")
        mp4 = args.out / f"{args.slug}.mp4"
        gif = args.out / f"{args.slug}.gif"
        poster = args.out / f"{args.slug}.jpg"
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
        print(f"{label:40s} {name:55s} p95 {metrics['tracking_p95_deg']:.3f} deg  fb {metrics['fb_share_pct']:.1f} %")
    for path in (mp4, gif, poster):
        print(path, f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
