"""Applies the scene calibration produced by scripts/calibrate_scene.py.

This is a module, not a runnable script. It exists so that every stage which
needs to interpret the calibration - the metrics pipeline and the demo renderer -
shares one implementation. Duplicating this logic would let the annotated video
and the reported numbers drift apart silently.

Responsibilities:
  - test whether a detection is inside the measurable road region
  - assign a detection to a lane
  - convert image motion into along-road distance

On distance and the focal length
--------------------------------
The calibration gives LATERAL metres-per-pixel: mpp(y) = h / (y - y_horizon),
fitted from car box widths. Vehicles, however, travel ALONG the road, and under
perspective the along-road scale is not the lateral one.

For a pinhole camera viewing a flat plane, along-road distance from the camera is

    D(y) = f * mpp(y)

for the camera's focal length f in pixels. f is not recoverable from this dataset
(no camera metadata, no surveyed reference length), so this module reports
along-road position in units of `metres / f` and calls it "raw". Two consequences:

  - RATIOS of raw distances and speeds are exact - f cancels. Anything expressed
    as a fraction of free-flow speed needs no anchor at all.
  - ABSOLUTE distances and speeds require choosing f. `Scene.anchor_focal_from_speed`
    derives it from an assumed free-flow speed, which must then be reported as an
    assumption rather than a measurement.

Using the lateral scale directly for along-road motion - the obvious shortcut -
would be wrong by a factor of f/(y - y_horizon), which varies about fourfold
across this scene's calibrated region, so it is deliberately not offered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "lanes_cctv052.json"


@dataclass
class Scene:
    frame_width: int
    frame_height: int
    y_min: float
    y_max: float
    num_lanes: int
    lane_polys: list[np.ndarray]
    roi_shift_left: float
    roi_shift_right: float
    patch_x0: float
    patch_x1: float
    scale_h: float
    scale_y_horizon: float
    focal_px: float | None = None  # set only by anchor_focal_from_speed()

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG) -> "Scene":
        cfg = json.loads(Path(path).read_text())
        y_min, y_max = cfg["calibrated_y_range"]
        box = cfg["patch_roi_box"]
        xs = [p[0] for p in box]
        return cls(
            frame_width=cfg["frame_width"],
            frame_height=cfg["frame_height"],
            y_min=float(y_min),
            y_max=float(y_max),
            num_lanes=cfg["num_lanes"],
            lane_polys=[np.array(p) for p in cfg["lane_polynomials"]],
            roi_shift_left=cfg["roi_edge_shift_px"]["left"],
            roi_shift_right=cfg["roi_edge_shift_px"]["right"],
            patch_x0=float(min(xs)),
            patch_x1=float(max(xs)),
            scale_h=cfg["scale_model"]["h"],
            scale_y_horizon=cfg["scale_model"]["y_horizon"],
        )

    # --------------------------------------------------------------- geometry

    def lane_x(self, lane: int, y):
        """x of a lane centerline at row y."""
        return np.polyval(self.lane_polys[lane], np.asarray(y, dtype=float))

    def roi_edges(self, y):
        """Left and right x bounds of the measurable road at row y, matching how
        derive_roi() built the stored polygon."""
        y = np.asarray(y, dtype=float)
        left = np.clip(self.lane_x(0, y) - self.roi_shift_left, self.patch_x0, self.patch_x1)
        right = np.clip(self.lane_x(-1, y) + self.roi_shift_right, self.patch_x0, self.patch_x1)
        return left, right

    def in_calibrated_rows(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return (y >= self.y_min) & (y <= self.y_max)

    def in_roi(self, x, y) -> np.ndarray:
        """Inside the measurable road region. Outside it the calibration says
        nothing, so callers must not lane-assign or speed-estimate there."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        left, right = self.roi_edges(y)
        return self.in_calibrated_rows(y) & (x >= left) & (x <= right)

    def lane_of(self, x, y):
        """Nearest lane centerline, or -1 outside the ROI. Returns an int array;
        scalars in, scalar-like array out."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        centers = np.stack([self.lane_x(i, y) for i in range(self.num_lanes)], axis=-1)
        nearest = np.argmin(np.abs(centers - x[:, None]), axis=-1)
        return np.where(self.in_roi(x, y), nearest, -1)

    # ------------------------------------------------------------------ scale

    def metres_per_pixel(self, y):
        """LATERAL scale at row y. Correct for widths and lateral offsets; not
        for along-road motion (see module docstring)."""
        return self.scale_h / (np.asarray(y, dtype=float) - self.scale_y_horizon)

    def along_road_raw(self, y):
        """Along-road position, in units of metres / f. Differences are meaningful;
        the absolute origin is not."""
        return self.metres_per_pixel(y)

    def along_road_raw_delta(self, y_from, y_to):
        """Along-road distance travelled between two rows, in metres / f."""
        return np.abs(self.along_road_raw(np.asarray(y_to)) - self.along_road_raw(np.asarray(y_from)))

    def raw_to_metres(self, raw):
        if self.focal_px is None:
            raise RuntimeError(
                "focal_px is unset: absolute distances need an anchor. "
                "Call anchor_focal_from_speed() first, or stay in raw units."
            )
        return np.asarray(raw, dtype=float) * self.focal_px

    def anchor_focal_from_speed(self, raw_speed_per_s: float, assumed_speed_kmh: float) -> float:
        """Pin the unknown focal length by asserting that a reference raw speed
        corresponds to a known real speed - in practice, that free-flowing traffic
        moves at about the posted limit.

        This makes absolute speeds an ASSUMPTION-ANCHORED estimate, not a
        measurement, and the reference case becomes true by construction. Speeds
        of other cases remain informative because they are measured relative to
        that anchor. Ratio-based results do not use this at all.
        """
        if raw_speed_per_s <= 0:
            raise ValueError("Reference raw speed must be positive")
        self.focal_px = (assumed_speed_kmh / 3.6) / raw_speed_per_s
        return self.focal_px

    # ------------------------------------------------------------- road extent

    def road_length_raw(self) -> float:
        """Along-road length of the calibrated region, in metres / f. Needed to
        express density per unit length."""
        return float(self.along_road_raw_delta(self.y_min, self.y_max))

    def lane_span_px(self, y) -> np.ndarray:
        """Distance between the outer lane centerlines at row y."""
        return np.abs(self.lane_x(-1, y) - self.lane_x(0, y))

    def mean_lane_width_m(self) -> float:
        """Mean spacing between adjacent lane centerlines in metres, using the
        lateral scale. A sanity check on the calibration: it should land near the
        3.7 m US standard lane width."""
        ys = np.linspace(self.y_min, self.y_max, 50)
        spacing_px = self.lane_span_px(ys) / (self.num_lanes - 1)
        return float(np.mean(spacing_px * self.metres_per_pixel(ys)))
