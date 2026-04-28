"""BVH locate and parse helpers for Kimodo retargeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from retarget_errors import MISSING_BVH, UNPARSEABLE_BVH, RetargetError


@dataclass(frozen=True)
class BvhJoint:
    name: str
    parent: str | None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: tuple[str, ...] = ()
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class BvhMotion:
    path: Path
    joints: dict[str, BvhJoint]
    root: str
    frames: list[list[float]]
    frame_time: float
    channel_order: list[tuple[str, str]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.frames) * self.frame_time


def locate_bvh(bundle_dir: Path, preferred_stem: str = "motion") -> Path:
    """Find the primary BVH artifact in a Kimodo output bundle."""

    bundle_dir = Path(bundle_dir)
    preferred = bundle_dir / f"{preferred_stem}.bvh"
    if preferred.is_file():
        return preferred
    matches = sorted(bundle_dir.rglob("*.bvh"))
    if matches:
        return matches[0]
    raise RetargetError(MISSING_BVH, f"No BVH artifact was found under {bundle_dir}.")


def parse_bvh(path: Path) -> BvhMotion:
    """Parse a BVH file enough for structural mapping and keyframe seams."""

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetargetError(UNPARSEABLE_BVH, f"Could not read BVH file {path}.") from exc

    try:
        return _parse_lines(path, lines)
    except RetargetError:
        raise
    except Exception as exc:
        raise RetargetError(UNPARSEABLE_BVH, f"BVH file {path} is structurally invalid: {exc}") from exc


def _parse_lines(path: Path, lines: list[str]) -> BvhMotion:
    joints: dict[str, dict] = {}
    stack: list[str] = []
    channel_order: list[tuple[str, str]] = []
    root: str | None = None
    motion_index = None

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        parts = stripped.split()
        if not parts:
            index += 1
            continue
        head = parts[0].upper()
        if head == "MOTION":
            motion_index = index
            break
        if head in {"ROOT", "JOINT"}:
            if len(parts) < 2:
                raise RetargetError(UNPARSEABLE_BVH, f"Missing joint name at line {index + 1}.")
            name = parts[1]
            parent = stack[-1] if stack else None
            if root is None:
                root = name
            joints[name] = {"parent": parent, "offset": (0.0, 0.0, 0.0), "channels": [], "children": []}
            if parent is not None:
                joints[parent]["children"].append(name)
            stack.append(name)
        elif head == "OFFSET" and stack:
            if len(parts) != 4:
                raise RetargetError(UNPARSEABLE_BVH, f"Invalid OFFSET at line {index + 1}.")
            joints[stack[-1]]["offset"] = tuple(float(value) for value in parts[1:4])
        elif head == "CHANNELS" and stack:
            count = int(parts[1])
            channels = parts[2:]
            if len(channels) != count:
                raise RetargetError(UNPARSEABLE_BVH, f"CHANNELS count mismatch at line {index + 1}.")
            joints[stack[-1]]["channels"] = channels
            channel_order.extend((stack[-1], channel) for channel in channels)
        elif stripped == "}":
            if stack:
                stack.pop()
        index += 1

    if motion_index is None or root is None or not channel_order:
        raise RetargetError(UNPARSEABLE_BVH, "BVH is missing hierarchy channels or MOTION section.")

    frames_count: int | None = None
    frame_time: float | None = None
    frame_rows: list[list[float]] = []
    for raw in lines[motion_index + 1:]:
        stripped = raw.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("frames:"):
            frames_count = int(stripped.split(":", 1)[1].strip())
        elif lower.startswith("frame time:"):
            frame_time = float(stripped.split(":", 1)[1].strip())
        else:
            values = [float(value) for value in stripped.split()]
            if len(values) != len(channel_order):
                raise RetargetError(UNPARSEABLE_BVH, "Motion frame channel count does not match hierarchy channels.")
            frame_rows.append(values)

    if frames_count is None or frame_time is None or frame_time <= 0:
        raise RetargetError(UNPARSEABLE_BVH, "BVH motion is missing Frames or Frame Time.")
    if frames_count != len(frame_rows):
        raise RetargetError(UNPARSEABLE_BVH, "BVH declared frame count does not match motion rows.")

    frozen_joints = {
        name: BvhJoint(
            name=name,
            parent=data["parent"],
            offset=data["offset"],
            channels=tuple(data["channels"]),
            children=tuple(data["children"]),
        )
        for name, data in joints.items()
    }
    return BvhMotion(path=path, joints=frozen_joints, root=root, frames=frame_rows, frame_time=frame_time, channel_order=channel_order)
