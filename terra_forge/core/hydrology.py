"""
Hydrology Engine:
River spline generation, channel carving, lake basin calculation, and water-proximity distance field.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import HydrologyConfig


class RiverSplineScalar(float):
    """
    Scalar float wrapper for river spline evaluation.
    Supports single-element indexing (e.g. `val[0]`), float casting,
    and scalar arithmetic without triggering NumPy deprecation warnings.
    """
    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def ndim(self) -> int:
        return 1

    def item(self) -> float:
        return float(self)

    def __getitem__(self, idx: int) -> float:
        if idx in (0, -1):
            return float(self)
        raise IndexError("RiverSplineScalar index out of range")

    def __len__(self) -> int:
        return 1

    def __sub__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(self) - float(other))

    def __rsub__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(other) - float(self))

    def __add__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(self) + float(other))

    def __radd__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(other) + float(self))

    def __mul__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(self) * float(other))

    def __rmul__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(other) * float(self))

    def __truediv__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(self) / float(other))

    def __rtruediv__(self, other: Any) -> RiverSplineScalar:
        return RiverSplineScalar(float(other) / float(self))

    def __neg__(self) -> RiverSplineScalar:
        return RiverSplineScalar(-float(self))


class HydrologyEngine:
    def __init__(self, hf: Heightfield2D, config: HydrologyConfig):
        self.hf = hf
        self.cfg = config
        self.dist_to_water = np.full((self.hf.res_y, self.hf.res_x), 999.0, dtype=np.float64)

    def evaluate_river_spline(
        self, t: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[RiverSplineScalar, RiverSplineScalar, RiverSplineScalar, RiverSplineScalar]:
        """
        Evaluates continuous multi-stage river spline (rx, ry, rz, rw) for parameter t in [0.0, 1.0].
        Dynamically reads self.cfg.river_stages to support arbitrary control points, tangents,
        widths, and depths per preset.
        Returns RiverSplineScalar tuple if t is a scalar, or numpy arrays if t is an array.
        """
        is_scalar = np.isscalar(t)
        t_arr = np.array([float(t)], dtype=np.float64) if is_scalar else np.asarray(t, dtype=np.float64)

        rx = np.zeros_like(t_arr)
        ry = np.zeros_like(t_arr)
        rz = np.zeros_like(t_arr)
        rw = np.zeros_like(t_arr)

        stages = self.cfg.river_stages or []
        if not stages:
            if is_scalar:
                return (
                    RiverSplineScalar(0.0),
                    RiverSplineScalar(0.0),
                    RiverSplineScalar(0.0),
                    RiverSplineScalar(0.0),
                )
            return rx, ry, rz, rw

        amp = self.cfg.river_meander_amplitude
        num_stages = len(stages)

        t_prev_end = 0.0
        for i, stage in enumerate(stages):
            t_start = float(stage.get("t_start", t_prev_end))
            t_end = float(stage.get("t_end", (i + 1.0) / num_stages if i < num_stages - 1 else 1.00))
            t_prev_end = t_end

            # Boolean mask for parameter domain within this stage
            if i == num_stages - 1:
                mask = (t_arr >= t_start) & (t_arr <= t_end)
            else:
                mask = (t_arr >= t_start) & (t_arr < t_end)

            if not np.any(mask):
                continue

            # Stage endpoints: (x, y, z, [w, [d]])
            s_pt = stage.get("start", (0.0, 0.0, 0.0, 4.0))
            e_pt = stage.get("end", (0.0, 0.0, 0.0, 4.0))

            x0, y0 = float(s_pt[0]), float(s_pt[1])
            z0 = float(s_pt[2])
            w0 = float(s_pt[3]) if len(s_pt) > 3 else 4.0

            x1, y1 = float(e_pt[0]), float(e_pt[1])
            z1 = float(e_pt[2])
            w1 = float(e_pt[3]) if len(e_pt) > 3 else 4.0

            stage_name = stage.get("name", "")

            # Dynamic binding to configured lake & sea levels
            if stage_name == "Central Lake Transit" or stage.get("is_lake_transit", False):
                z0 = float(self.cfg.lake_level)
                z1 = float(self.cfg.lake_level)
            elif stage_name == "Outlet Gorge & Bay" or stage.get("is_outlet", False):
                z0 = float(self.cfg.lake_level)
                z1 = float(self.cfg.sea_level)
            elif i == num_stages - 1 and abs(z1 - 0.0) < 1e-4 and abs(self.cfg.sea_level - 0.0) > 1e-4:
                z1 = float(self.cfg.sea_level)

            span = max(1e-7, t_end - t_start)
            u = (t_arr[mask] - t_start) / span
            su = 3.0 * u**2 - 2.0 * u**3

            # Check for specific named presets to maintain exact baseline calibrations
            if stage_name == "Mountain Cascades":
                rx[mask] = x0 + (x1 - x0) * u + (2.5 * amp) * np.sin(np.pi * u)
                ry[mask] = y0 + (y1 - y0) * u
                rz[mask] = z0 - (z0 - z1) * su
                rw[mask] = w0 + (w1 - w0) * u
            elif stage_name == "Valley Meander":
                rx[mask] = x0 + (x1 - x0) * u - (8.0 * amp) * np.sin(2.0 * np.pi * u)
                ry[mask] = y0 + (y1 - y0) * u + (4.0 * amp) * np.sin(np.pi * u)
                rz[mask] = z0 - (z0 - z1) * su
                rw[mask] = w0 + (w1 - w0) * u
            elif stage_name == "Central Lake Transit":
                rx[mask] = x0 + (x1 - x0) * u
                ry[mask] = y0 + (y1 - y0) * u
                rz[mask] = z0
                rw[mask] = w0 + (w1 - w0) * u
            elif stage_name == "Outlet Gorge & Bay":
                rx[mask] = x0 + (x1 - x0) * u + (6.0 * amp) * np.sin(np.pi * u)
                ry[mask] = y0 + (y1 - y0) * u
                rz[mask] = z0 - (z0 - z1) * su
                rw[mask] = w0 + (w1 - w0) * u
            else:
                # Generalized parametric curve supporting arbitrary tangents and meander oscillations
                if "tangent_in" in stage and "tangent_out" in stage:
                    t_in = stage["tangent_in"]
                    t_out = stage["tangent_out"]
                    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
                    h10 = u**3 - 2.0 * u**2 + u
                    h01 = -2.0 * u**3 + 3.0 * u**2
                    h11 = u**3 - u**2
                    rx[mask] = h00 * x0 + h10 * t_in[0] + h01 * x1 + h11 * t_out[0]
                    ry[mask] = h00 * y0 + h10 * t_in[1] + h01 * y1 + h11 * t_out[1]
                    rz[mask] = h00 * z0 + h10 * t_in[2] + h01 * z1 + h11 * t_out[2]
                else:
                    dx = x1 - x0
                    dy = y1 - y0
                    seg_len = math.sqrt(dx**2 + dy**2) + 1e-7
                    nx = -dy / seg_len
                    ny = dx / seg_len

                    stage_amp = float(stage.get("meander_amp", 1.0)) * amp
                    cycles = float(stage.get("meander_cycles", 1.0))
                    disp = stage_amp * np.sin(cycles * np.pi * u)

                    rx[mask] = x0 + dx * u + disp * nx
                    ry[mask] = y0 + dy * u + disp * ny
                    rz[mask] = z0 - (z0 - z1) * su

                rw[mask] = w0 + (w1 - w0) * u

        if is_scalar:
            return (
                RiverSplineScalar(rx[0]),
                RiverSplineScalar(ry[0]),
                RiverSplineScalar(rz[0]),
                RiverSplineScalar(rw[0]),
            )

        return rx, ry, rz, rw

    def evaluate_flow_velocity(
        self, t: np.ndarray | float, v_base: float = 2.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[RiverSplineScalar, RiverSplineScalar, RiverSplineScalar]:
        """
        Calculates 3D flow velocity vector (vx, vy, vz) along the river spline.
        Downhill potential gradient (-dz/dt) accelerates flow velocity over cascades
        and decelerates in gentle valley meanders.
        """
        is_scalar = np.isscalar(t)
        eps = 1e-4
        t_val = float(t) if is_scalar else np.asarray(t, dtype=np.float64)

        t_minus = np.clip(t_val - eps, 0.0, 1.0)
        t_plus = np.clip(t_val + eps, 0.0, 1.0)

        rx_m, ry_m, rz_m, _ = self.evaluate_river_spline(t_minus)
        rx_p, ry_p, rz_p, _ = self.evaluate_river_spline(t_plus)

        dt = np.maximum(1e-7, (t_plus - t_minus))
        tx = (rx_p - rx_m) / dt
        ty = (ry_p - ry_m) / dt
        tz = (rz_p - rz_m) / dt

        tangent_len = np.sqrt(tx**2 + ty**2 + tz**2) + 1e-7
        # Downhill gradient acceleration: -tz represents drop rate
        speed = v_base * np.sqrt(np.maximum(0.1, -tz))

        vx = speed * (tx / tangent_len)
        vy = speed * (ty / tangent_len)
        vz = speed * (tz / tangent_len)

        if is_scalar:
            def _to_scalar(val: Any) -> RiverSplineScalar:
                if hasattr(val, "flat"):
                    return RiverSplineScalar(float(val.flat[0]))
                return RiverSplineScalar(float(val))

            return (
                _to_scalar(vx),
                _to_scalar(vy),
                _to_scalar(vz),
            )
        return vx, vy, vz

    def compute_flow_velocity_field(self, v_base: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates 2D velocity vector fields (Vx, Vy) across the heightfield grid.
        Within river ribbons, vectors align with channel trajectory; outside,
        they align with terrain drainage gradient.
        """
        ny, nx = self.hf.res_y, self.hf.res_x
        vx_field = np.zeros((ny, nx), dtype=np.float64)
        vy_field = np.zeros((ny, nx), dtype=np.float64)

        if not self.cfg.river_enabled:
            return vx_field, vy_field

        t_samples = np.linspace(0.0, 1.0, 200)
        rx, ry, _, rw = self.evaluate_river_spline(t_samples)
        vx_spline, vy_spline, _ = self.evaluate_flow_velocity(t_samples, v_base=v_base)

        for i in range(len(t_samples) - 1):
            cx, cy = rx[i], ry[i]
            width = rw[i]
            r_inf = width * 1.2

            gx_min, gy_min = self.hf.world_to_grid(cx - r_inf, cy - r_inf)
            gx_max, gy_max = self.hf.world_to_grid(cx + r_inf, cy + r_inf)

            ix0 = max(0, int(gx_min))
            ix1 = min(nx, int(gx_max) + 1)
            iy0 = max(0, int(gy_min))
            iy1 = min(ny, int(gy_max) + 1)

            if ix1 <= ix0 or iy1 <= iy0:
                continue

            sub_x = self.hf.X[iy0:iy1, ix0:ix1]
            sub_y = self.hf.Y[iy0:iy1, ix0:ix1]
            dist = np.sqrt((sub_x - cx)**2 + (sub_y - cy)**2)
            mask = dist < r_inf

            if np.any(mask):
                weight = 1.0 - (dist[mask] / r_inf)
                vx_field[iy0:iy1, ix0:ix1][mask] = vx_spline[i] * weight
                vy_field[iy0:iy1, ix0:ix1][mask] = vy_spline[i] * weight

        return vx_field, vy_field

    def carve_river_channel(self, num_samples: int = 250) -> None:
        """Carves a parabolic river bed along the river spline into the heightfield."""
        if not self.cfg.river_enabled:
            return

        t_samples = np.linspace(0.0, 1.0, num_samples)
        rx, ry, rz, rw = self.evaluate_river_spline(t_samples)

        carve_depth = self.cfg.river_carve_depth

        for i in range(num_samples):
            cx, cy, cz, width = rx[i], ry[i], rz[i], rw[i]
            influence_radius = width * 1.5

            # Find local grid bounding box
            gx_min, gy_min = self.hf.world_to_grid(cx - influence_radius, cy - influence_radius)
            gx_max, gy_max = self.hf.world_to_grid(cx + influence_radius, cy + influence_radius)

            ix0 = max(0, int(gx_min))
            ix1 = min(self.hf.res_x, int(gx_max) + 1)
            iy0 = max(0, int(gy_min))
            iy1 = min(self.hf.res_y, int(gy_max) + 1)

            if ix1 <= ix0 or iy1 <= iy0:
                continue

            sub_x = self.hf.X[iy0:iy1, ix0:ix1]
            sub_y = self.hf.Y[iy0:iy1, ix0:ix1]
            dist_sq = (sub_x - cx)**2 + (sub_y - cy)**2
            dist = np.sqrt(dist_sq)

            mask = dist < influence_radius
            if np.any(mask):
                norm_d = dist[mask] / influence_radius
                # Parabolic trench
                trench = (1.0 - norm_d**2) * carve_depth
                target_bed = cz - trench
                self.hf.Z[iy0:iy1, ix0:ix1][mask] = np.minimum(self.hf.Z[iy0:iy1, ix0:ix1][mask], target_bed)

    def compute_water_distance_field(self) -> np.ndarray:
        """
        Computes the Euclidean distance from every grid point to the nearest water body
        (Central Lake or River spline). Used for biome ecological scattering.
        """
        # Distance to Central Lake
        lx, ly = self.cfg.lake_center
        if self.cfg.lake_radius > 0.0:
            dist_to_lake_center = np.sqrt((self.hf.X - lx)**2 + (self.hf.Y - ly)**2)
            dist_lake = np.maximum(0.0, dist_to_lake_center - self.cfg.lake_radius)
        else:
            dist_lake = np.full_like(self.hf.Z, 999.0)

        # Distance to River Spline
        if self.cfg.river_enabled:
            t_samples = np.linspace(0.0, 1.0, 100)
            rx, ry, _, rw = self.evaluate_river_spline(t_samples)

            dist_river = np.full_like(self.hf.Z, 999.0)
            for i in range(len(t_samples)):
                d_spline = np.sqrt((self.hf.X - rx[i])**2 + (self.hf.Y - ry[i])**2) - (rw[i] * 0.5)
                dist_river = np.minimum(dist_river, np.maximum(0.0, d_spline))

            self.dist_to_water = np.minimum(dist_lake, dist_river)
        else:
            self.dist_to_water = dist_lake

        return self.dist_to_water
