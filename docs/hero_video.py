"""Hero video for the report: the same belt-impact rung under three controllers, side by side, with the force trace.

    MUJOCO_GL=egl env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test \
        python docs/hero_video.py --out /home/bam/robot_controllers/reports/media/robot-controllers-2026-11-12

Every frame comes from a real bench episode (`run_episode`): the joint angles are the plant's logged samples and the
force is the logged belt contact force. Rendering only adds a light, a floor and stripes on the belt so its motion
shows; it never changes the numbers.
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
from robot_controllers.bench.loop import run_episode  # noqa: E402
from robot_controllers.bench.rungs import rung as rung_by_id  # noqa: E402
from robot_controllers.registry import controller  # noqa: E402

RUNG = "R3.BeltImpact@0.10m/s"
CONTROLLERS = ("joint_trajectory_controller", "joint_impedance_board", "joint_impedance_host")
COLOURS = ("#f97316", "#22c55e", "#60a5fa")
PANEL_W, PANEL_H, PLOT_H = 520, 420, 300
FPS = 30
SLOWDOWN = 4.0  # WHY slow motion: the impact transient is ~30 ms; at real speed it is one frame


def render_model() -> mujoco.MjModel:
    """The bench's belt MJCF plus a light, a floor and belt stripes (visual geoms only, no contact)."""
    xml = build_mjcf(belt=True, arm=ARM)
    stripes = "".join(
        f'<geom type="box" pos="{x:.3f} 0 0.0505" size="0.008 0.148 0.0005" contype="0" conaffinity="0" '
        f'rgba="0.85 0.85 0.85 1"/>'
        for x in np.arange(-1.5, 1.5, 0.05)
    )
    xml = xml.replace('rgba="0.2 0.2 0.2 1"/>', 'rgba="0.18 0.18 0.2 1"/>' + stripes, 1)
    xml = xml.replace(
        "<worldbody>",
        '<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="2048"/></visual>'
        "<worldbody>"
        '<light pos="0.3 -1.2 1.4" dir="0 0.6 -0.8" diffuse="0.9 0.9 0.9" castshadow="true"/>'
        '<geom type="plane" size="2 2 0.01" pos="0 0 0" rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>',
        1,
    )
    return mujoco.MjModel.from_xml_string(xml)


def episode(name: str):
    rung = rung_by_id(RUNG)
    log = run_episode(controller(name), rung, seed=0)
    return rung, log


def panel_frames(model, log, times, belt_speed):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.27, 0.0, 0.27)
    camera.distance = 0.78
    camera.azimuth = 90.0
    camera.elevation = -8.0
    frames = []
    for t in times:
        data.qpos[0] = np.interp(t, log.t, log.q[:, 0])
        data.qpos[1] = np.interp(t, log.t, log.q[:, 1])
        data.qpos[2] = belt_speed * t
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def force_plot(logs, times, t_now, touch_window):
    fig, ax = plt.subplots(figsize=(3 * PANEL_W / 100, PLOT_H / 100), dpi=100)
    fig.patch.set_facecolor("#0b0b0c")
    ax.set_facecolor("#0b0b0c")
    for (name, log), colour in zip(logs, COLOURS, strict=True):
        ax.plot(log.t, log.contact_force[:, 2], color=colour, lw=1.6, label=name)
    if touch_window:
        ax.axvspan(*touch_window, color="#facc15", alpha=0.12, lw=0)
    ax.axvline(t_now, color="#e4e4e7", lw=1.0)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("time (s)", color="#a1a1aa")
    ax.set_ylabel("belt normal force (N)", color="#a1a1aa")
    ax.tick_params(colors="#a1a1aa")
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.legend(loc="upper left", fontsize=9, facecolor="#18181b", edgecolor="#3f3f46", labelcolor="#e4e4e7")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = render_model()
    rung = rung_by_id(RUNG)
    logs = [(name, episode(name)[1]) for name in CONTROLLERS]
    duration = min(float(log.t[-1]) for _, log in logs)
    times = np.arange(0.0, duration, 1.0 / (FPS * SLOWDOWN))
    belt_speed = float(rung.scene.belt_speed_m_s)
    touch = [log.t[np.argmax(log.in_contact)] for _, log in logs if log.in_contact.any()]
    touch_window = (min(touch), min(touch) + 0.030) if touch else None
    panels = [panel_frames(model, log, times, belt_speed) for _, log in logs]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()

    with tempfile.TemporaryDirectory() as tmp:
        for index, t in enumerate(times):
            row = np.concatenate([frames[index] for frames in panels], axis=1)
            plot = force_plot(logs, times, t, touch_window)
            plot = np.asarray(Image.fromarray(plot).resize((row.shape[1], PLOT_H)))
            frame = Image.fromarray(np.concatenate([row, plot], axis=0))
            draw = ImageDraw.Draw(frame)
            for column, ((name, log), colour) in enumerate(zip(logs, COLOURS, strict=True)):
                force = float(np.interp(t, log.t, log.contact_force[:, 2]))
                draw.text((column * PANEL_W + 14, 10), name, fill=colour, font=font)
                draw.text((column * PANEL_W + 14, 34), f"{force:5.1f} N on the belt", fill="#e4e4e7", font=small)
            draw.text((14, PANEL_H - 26), f"{RUNG} · t = {t:.3f} s · {SLOWDOWN:g}x slow motion", fill="#a1a1aa",
                      font=small)
            frame.save(Path(tmp) / f"f{index:05d}.png")
        mp4 = args.out / "hero-belt-impact.mp4"
        gif = args.out / "hero-belt-impact.gif"
        poster = args.out / "hero-belt-impact.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=12,scale=780:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                        str(gif)], check=True)
        middle = len(times) // 2
        Image.open(Path(tmp) / f"f{int(np.searchsorted(times, touch_window[0] + 0.01) if touch_window else middle):05d}.png"
                   ).convert("RGB").save(poster, quality=85)
    for path in (mp4, gif, poster):
        print(path, f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
