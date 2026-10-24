"""
2D Heightfield Representation & Mathematical Topography Synthesis.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from terra_forge.schema.map_config import MapConfig



def _perlin_2d(X: np.ndarray, Y: np.ndarray, freq: float, seed: int = 42) -> np.ndarray:
    """
    Vectorized 2D Perlin gradient noise with C2 quintic Hermite interpolation.
    Returns array with values smoothly in [-1.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    gx = X * freq
    gy = Y * freq
    x0 = np.floor(gx).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(gy).astype(np.int32)
    y1 = y0 + 1
    fx = gx - x0
    fy = gy - y0

    # Quintic Hermite fade curve: 6t^5 - 15t^4 + 10t^3 (C2 continuity)
    u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)

    # Gradient permutation table
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)

    grads = np.array([
        [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]
    ], dtype=np.float64) / math.sqrt(2.0)

    h00 = (p[x0 % 256] + y0) % 256
    h10 = (p[x1 % 256] + y0) % 256
    h01 = (p[x0 % 256] + y1) % 256
    h11 = (p[x1 % 256] + y1) % 256

    g00 = grads[p[h00] % 8]
    g10 = grads[p[h10] % 8]
    g01 = grads[p[h01] % 8]
    g11 = grads[p[h11] % 8]

    d00 = g00[:, :, 0] * fx + g00[:, :, 1] * fy
    d10 = g10[:, :, 0] * (fx - 1.0) + g10[:, :, 1] * fy
    d01 = g01[:, :, 0] * fx + g01[:, :, 1] * (fy - 1.0)
    d11 = g11[:, :, 0] * (fx - 1.0) + g11[:, :, 1] * (fy - 1.0)

    nx0 = (1.0 - u) * d00 + u * d10
    nx1 = (1.0 - u) * d01 + u * d11
    return (1.0 - v) * nx0 + v * nx1


class Heightfield2D:
    def __init__(self, width: float = 160.0, depth: float = 160.0, resolution: int = 256):
        self.width = float(width)
        self.depth = float(depth)
        self.res_x = int(resolution)
        self.res_y = int(resolution)
        self.dx = self.width / (self.res_x - 1)
        self.dy = self.depth / (self.res_y - 1)

        # Coordinate axes: centered at (0, 0)
        self.x_coords = np.linspace(-self.width / 2.0, self.width / 2.0, self.res_x, dtype=np.float64)
        self.y_coords = np.linspace(-self.depth / 2.0, self.depth / 2.0, self.res_y, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        # Elevation grid Z[y, x]
        self.Z = np.zeros((self.res_y, self.res_x), dtype=np.float64)
        # Geological sediment layer (accumulated deposit from hydraulic erosion)
        self.sediment = np.zeros((self.res_y, self.res_x), dtype=np.float64)
        # Thermal scree / talus accumulation layer (accumulated rockfall debris)
        self.scree_talus = np.zeros((self.res_y, self.res_x), dtype=np.float64)
        # Cumulative water flow depth flux map
        self.water_depth = np.zeros((self.res_y, self.res_x), dtype=np.float64)


    def world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Converts world coordinates (x, y) to floating grid indices (gx, gy)."""
        gx = (x + self.width / 2.0) / self.dx
        gy = (y + self.depth / 2.0) / self.dy
        return gx, gy

    def grid_to_world(self, gx: float, gy: float) -> Tuple[float, float]:
        """Converts grid indices (gx, gy) to world coordinates (x, y)."""
        x = -self.width / 2.0 + gx * self.dx
        y = -self.depth / 2.0 + gy * self.dy
        return x, y

    def sample_bilinear(self, x: float, y: float) -> float:
        """Sample elevation at arbitrary continuous world coordinates with bilinear interpolation."""
        gx, gy = self.world_to_grid(x, y)
        gx = np.clip(gx, 0, self.res_x - 1.0001)
        gy = np.clip(gy, 0, self.res_y - 1.0001)

        x0, y0 = int(gx), int(gy)
        x1, y1 = x0 + 1, y0 + 1
        fx = gx - x0
        fy = gy - y0

        z00 = self.Z[y0, x0]
        z10 = self.Z[y0, x1]
        z01 = self.Z[y1, x0]
        z11 = self.Z[y1, x1]

        return float((1 - fx) * (1 - fy) * z00 + fx * (1 - fy) * z10 + (1 - fx) * fy * z01 + fx * fy * z11)

    def compute_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes dZ/dx and dZ/dy across the grid using central differences."""
        dz_dy, dz_dx = np.gradient(self.Z, self.dy, self.dx)
        return dz_dx, dz_dy

    def compute_normals(self) -> np.ndarray:
        """
        Computes 3D normal vectors (Nx, Ny, Nz) of shape (res_y, res_x, 3).
        Surface normal is proportional to (-dZ/dx, -dZ/dy, 1.0).
        """
        dz_dx, dz_dy = self.compute_gradients()
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(self.Z)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
        return normals

    def compute_slope_degrees(self) -> np.ndarray:
        """Computes slope angle in degrees [0, 90] across the grid."""
        normals = self.compute_normals()
        nz = np.clip(normals[:, :, 2], 0.0, 1.0)
        slope_rad = np.arccos(nz)
        return np.degrees(slope_rad)

    def add_peak(
        self,
        center_x: float,
        center_y: float,
        height: float,
        radius: float,
        shape: str = "horn",
        ridge_angle_deg: float = 0.0,
    ) -> None:
        """
        Synthesizes a mountain massif, horn, ridge, plateau, or volcanic dome.
        """
        dx = self.X - center_x
        dy = self.Y - center_y
        dist = np.sqrt(dx**2 + dy**2)

        if shape == "horn":
            # Alpine Matterhorn-style pyramidal horn with radiating knife-edge aretes and glacial cirques
            norm_dist = np.clip(dist / radius, 0.0, 1.0)
            base_elevation = height * ((1.0 - norm_dist) ** 2.0)

            angle = np.arctan2(dy, dx) - math.radians(ridge_angle_deg)
            num_aretes = 3.0  # Trihedral Matterhorn pyramid

            cos_val = np.cos(num_aretes * angle)
            # Sharp knife-edge arete spine ribs (narrow positive crests)
            sharp_aretes = np.maximum(0.0, cos_val) ** 0.6
            # Glacial cirque hollows (concave scoops between aretes)
            cirque_hollow = np.maximum(0.0, -cos_val) ** 1.2
            cirque_envelope = (np.sin(math.pi * norm_dist) ** 1.4) * np.clip(norm_dist * 2.0, 0.0, 1.0)

            elevation = base_elevation * (1.0 + 0.35 * sharp_aretes - 0.15 * cirque_hollow * cirque_envelope)
            self.Z += np.maximum(0.0, elevation)

        elif shape == "ridge" or shape == "arete":
            # Razor knife-edge arete rock spine with transverse cusp profile
            rad = math.radians(ridge_angle_deg)
            along_ridge = dx * math.cos(rad) + dy * math.sin(rad)
            across_ridge = -dx * math.sin(rad) + dy * math.cos(rad)

            along_norm = np.clip(np.abs(along_ridge) / (radius * 1.5), 0.0, 1.0)
            across_norm = np.clip(np.abs(across_ridge) / (radius * 0.35), 0.0, 1.0)

            # Transverse cusp: sharp apex profile
            transverse_profile = (1.0 - across_norm) ** 2.5
            longitudinal_profile = (1.0 - along_norm) ** 1.8

            # Secondary craggy jaggedness along spine
            spine_jagged = 1.0 + 0.12 * np.cos(0.4 * along_ridge)
            elevation = height * longitudinal_profile * transverse_profile * spine_jagged
            self.Z += np.maximum(0.0, elevation)

        elif shape == "plateau":
            # Flat top plateau with steep talus rim
            norm_dist = np.clip(dist / radius, 0.0, 1.0)
            # Smoothstep cliff edge
            cliff = 1.0 / (1.0 + np.exp(12.0 * (norm_dist - 0.7)))
            self.Z += height * cliff

        elif shape == "dome":
            # Rounded volcanic shield dome
            norm_dist = np.clip(dist / radius, 0.0, 1.0)
            elevation = height * np.cos(norm_dist * (math.pi / 2.0))**1.4
            self.Z += elevation

        elif shape == "cone":
            # Linear geometric cone: height * max(0.0, 1.0 - dist / radius)
            norm_dist = np.clip(dist / radius, 0.0, 1.0)
            elevation = height * (1.0 - norm_dist)
            self.Z += elevation

    def carve_lake_basin(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        water_level: float,
        lake_depth: float,
        berm_height: float = 1.2,
    ) -> None:
        """
        Carves a bathymetric lake basin with natural shelves and deep trenches,
        and constructs a surrounding retaining berm / moraine rim to ensure
        the water is physically contained and does not float above surrounding ground.
        """
        dx = self.X - center_x
        dy = self.Y - center_y
        dist = np.sqrt(dx**2 + dy**2)
        norm_dist = dist / radius

        # If water_level is elevated above sea level, construct retaining berm
        if water_level > 0.0 and berm_height > 0.0:
            # 1. Outer berm falloff (from 1.15 * R to 1.5 * R)
            outer_mask = (norm_dist > 1.15) & (norm_dist <= 1.50)
            if np.any(outer_mask):
                u_out = (norm_dist[outer_mask] - 1.15) / 0.35
                s_out = 1.0 - (3.0 * u_out**2 - 2.0 * u_out**3)
                z_outer = (water_level + berm_height) * s_out
                self.Z[outer_mask] = np.maximum(self.Z[outer_mask], z_outer)

            # 2. Berm crest (from 0.95 * R to 1.15 * R)
            crest_mask = (norm_dist >= 0.95) & (norm_dist <= 1.15)
            if np.any(crest_mask):
                z_crest = water_level + berm_height
                self.Z[crest_mask] = np.maximum(self.Z[crest_mask], z_crest)

            # 3. Inside basin (< 0.95 * R): smoothstep from deep bottom to berm crest
            inside = norm_dist < 0.95
            if np.any(inside):
                nd = norm_dist[inside] / 0.95
                s = 3.0 * nd**2 - 2.0 * nd**3
                target_z = (water_level - lake_depth) + (lake_depth + berm_height) * s
                self.Z[inside] = target_z

        else:
            # Standard flat-sea basin
            inside = norm_dist < 1.0
            if np.any(inside):
                nd = norm_dist[inside]
                depth_curve = 1.0 - (3.0 * nd**2 - 2.0 * nd**3)
                target_z = water_level - lake_depth * depth_curve
                self.Z[inside] = np.minimum(self.Z[inside], target_z)


    def add_perlin_octaves(self, seed: int = 42, octaves: int = 4, persistence: float = 0.5, scale: float = 40.0, amplitude: float = 4.0) -> None:
        """Adds procedural pseudo-fractal noise to create organic micro-roughness."""
        rng = np.random.default_rng(seed)
        noise_total = np.zeros_like(self.Z)
        current_amp = amplitude
        current_freq = 1.0 / scale

        for _ in range(octaves):
            # Generate random frequency grid
            grid_w = max(4, int(self.width * current_freq))
            grid_h = max(4, int(self.depth * current_freq))
            coarse = rng.standard_normal((grid_h, grid_w))

            # Pure numpy bilinear upsample to (res_y, res_x)
            y_indices = np.linspace(0, grid_h - 1, self.res_y)
            x_indices = np.linspace(0, grid_w - 1, self.res_x)
            x0 = np.floor(x_indices).astype(int)
            x1 = np.clip(x0 + 1, 0, grid_w - 1)
            y0 = np.floor(y_indices).astype(int)
            y1 = np.clip(y0 + 1, 0, grid_h - 1)
            wx = (x_indices - x0)[np.newaxis, :]
            wy = (y_indices - y0)[:, np.newaxis]

            c00 = coarse[y0[:, np.newaxis], x0[np.newaxis, :]]
            c10 = coarse[y0[:, np.newaxis], x1[np.newaxis, :]]
            c01 = coarse[y1[:, np.newaxis], x0[np.newaxis, :]]
            c11 = coarse[y1[:, np.newaxis], x1[np.newaxis, :]]
            upsampled = (1 - wy) * ((1 - wx) * c00 + wx * c10) + wy * ((1 - wx) * c01 + wx * c11)

            noise_total += upsampled * current_amp
            current_amp *= persistence
            current_freq *= 2.0

        self.Z += noise_total

    def apply_strata_folding(
        self,
        amplitude: float = 6.0,
        wavelength: float = 45.0,
        strike_angle_deg: float = 35.0,
        plunge_wavelength: float = 120.0,
        harmonic_weight: float = 0.25,
        domain_warp_amp: float = 8.0,
        domain_warp_freq: float = 0.04,
    ) -> None:
        """
        Synthesizes tectonic strata folding (anticlines and synclines) with domain warping.

        Args:
            amplitude: Peak height in meters of the folding wave.
            wavelength: Cross-strike distance between successive anticline crests in meters.
            strike_angle_deg: Azimuthal orientation of the fold axes in degrees.
            plunge_wavelength: Along-strike wavelength of plunging fold hinges (<= 0 to disable).
            harmonic_weight: Amplitude ratio of secondary harmonic overtone for asymmetric fold vergence.
            domain_warp_amp: Lateral displacement amplitude for organic domain warping in meters.
            domain_warp_freq: Spatial frequency of domain warping distortion (1/m).
        """
        if amplitude == 0.0 or wavelength <= 0.0:
            return

        rad = math.radians(strike_angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # 1. Tectonic Domain Warping
        if domain_warp_amp > 0.0 and domain_warp_freq > 0.0:
            warp_phase_x = 2.0 * math.pi * domain_warp_freq * (self.Y * cos_a - self.X * sin_a)
            warp_phase_y = 2.0 * math.pi * domain_warp_freq * (self.X * cos_a + self.Y * sin_a)
            x_warped = self.X + domain_warp_amp * np.sin(warp_phase_x)
            y_warped = self.Y + domain_warp_amp * np.cos(warp_phase_y)
        else:
            x_warped = self.X
            y_warped = self.Y

        # 2. Projection to Cross-Strike (u) and Along-Strike (v) Coordinates
        u = x_warped * cos_a + y_warped * sin_a
        v = -x_warped * sin_a + y_warped * cos_a

        # 3. Asymmetric Waveform
        k1 = 2.0 * math.pi / wavelength
        wave = np.sin(k1 * u)
        if harmonic_weight > 0.0:
            psi = math.pi / 6.0  # 30-degree phase shift produces asymmetrical fold vergence
            wave += harmonic_weight * np.sin(2.0 * k1 * u + psi)

        # 4. Plunging Fold Hinge Modulation
        if plunge_wavelength > 0.0:
            kp = 2.0 * math.pi / plunge_wavelength
            mod_plunge = 1.0 + 0.15 * np.cos(kp * v)
            wave *= mod_plunge

        self.Z += amplitude * wave

    def add_strata_fold(
        self,
        amplitude: float = 6.0,
        wavelength: float = 45.0,
        strike_angle_deg: float = 35.0,
        plunge_wavelength: float = 120.0,
        harmonic_weight: float = 0.25,
        domain_warp_amp: float = 0.0,
        domain_warp_freq: float = 0.04,
    ) -> None:
        """Alias / convenience wrapper for apply_strata_folding."""
        self.apply_strata_folding(
            amplitude=amplitude,
            wavelength=wavelength,
            strike_angle_deg=strike_angle_deg,
            plunge_wavelength=plunge_wavelength,
            harmonic_weight=harmonic_weight,
            domain_warp_amp=domain_warp_amp,
            domain_warp_freq=domain_warp_freq,
        )

    def apply_canyon_terraces(
        self,
        terrace_height: float = 6.0,
        terrace_sharpness: float = 6.0,
        blend: float = 0.85,
        noise_modulation: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Applies a continuous hyperbolic tangent staircase terrace transfer function
        to simulate differential weathering of layered sedimentary rock (canyon terraces).

        Args:
            terrace_height: Vertical thickness of one strata layer in meters.
            terrace_sharpness: Sharpness parameter k governing cliff verticality.
            blend: Blend weight [0.0, 1.0] between original terrain and terraced profile.
            noise_modulation: Amplitude of spatial variation in layer thickness to break horizontal monotony.
        """
        if "step_height" in kwargs:
            terrace_height = float(kwargs["step_height"])
        if "sharpness" in kwargs:
            terrace_sharpness = float(kwargs["sharpness"])
        if "blend_weight" in kwargs:
            blend = float(kwargs["blend_weight"])

        if terrace_height <= 0.0 or blend <= 0.0:
            return

        blend = float(np.clip(blend, 0.0, 1.0))
        k = max(0.5, float(terrace_sharpness))

        # Effective stratum thickness with optional low-frequency spatial modulation
        if noise_modulation > 0.0:
            H = terrace_height * (1.0 + noise_modulation * np.sin(0.04 * self.X) * np.cos(0.04 * self.Y))
            H = np.maximum(H, 0.5)
        else:
            H = terrace_height

        # Normalized layer phase u in [0, 1)
        u = (self.Z % H) / H

        # Continuous hyperbolic tangent step function S(u, k)
        denom = 2.0 * math.tanh(0.5 * k)
        step = np.tanh(k * (u - 0.5)) / denom + 0.5

        # Reconstructed staircase elevation
        z_terraced = np.floor(self.Z / H) * H + H * step

        # Smooth blend with original topography
        self.Z = (1.0 - blend) * self.Z + blend * z_terraced

    def apply_tectonic_scarp(
        self,
        fault_angle_deg: float = -25.0,
        displacement_throw: float = 14.0,
        scarp_sharpness: float = 10.0,
        fault_offset: float = 12.0,
        fault_length: float = 0.0,
    ) -> None:
        """
        Synthesizes a tectonic fault scarp with sheer vertical displacement along a fault line.

        Args:
            fault_angle_deg: Azimuthal orientation of the fault strike in degrees.
            displacement_throw: Total vertical throw displacement in meters.
            scarp_sharpness: Sigmoid parameter k governing cliff wall verticality.
            fault_offset: Perpendicular offset from diorama center in meters.
            fault_length: Along-strike fault length with tapering at tips (<= 0 for regional fault).
        """
        if displacement_throw == 0.0:
            return

        rad = math.radians(fault_angle_deg)
        cos_f = math.cos(rad)
        sin_f = math.sin(rad)

        # Signed perpendicular distance from the fault line: d = -X*sin + Y*cos - offset
        d_perp = -self.X * sin_f + self.Y * cos_f - fault_offset

        # Sigmoid transition: 1 / (1 + exp(-clip(k * d, -60, 60)))
        k = max(0.1, float(scarp_sharpness))
        kd = np.clip(k * d_perp, -60.0, 60.0)
        sigmoid = 1.0 / (1.0 + np.exp(-kd))

        delta_z = displacement_throw * (sigmoid - 0.5)

        # Optional along-strike tapering for finite faults
        if fault_length > 0.0:
            s_along = self.X * cos_f + self.Y * sin_f
            taper = np.maximum(0.0, 1.0 - (s_along / (0.5 * fault_length)) ** 2)
            delta_z *= taper

        self.Z += delta_z

    def apply_coastal_profile(
        self,
        sea_level: float = 0.0,
        wave_notch_depth: float = 1.8,
        wave_notch_envelope: float = 1.2,
        beach_slope_coefficient: float = 0.12,
        berm_elevation: float = 1.2,
        beach_width: float = 15.0,
        shelf_transition_dist: float = 35.0,
        deep_water_depth: float = 12.0,
        coastline_angle_deg: Optional[float] = None,
        coastline_offset: float = 0.0,
    ) -> None:
        """
        Applies coastal geomorphology profiles:
        1. Dean's equilibrium beach profile (h(y) = A * y^(2/3)) transitioning to deep bathymetry.
        2. Supratidal sandy beach berm.
        3. Intertidal wave-cut notches undercutting coastal cliffs at sea level.
        """
        # Part 1: Directional Coastline Synthesis (if coastline_angle_deg is specified)
        if coastline_angle_deg is not None:
            rad = math.radians(coastline_angle_deg)
            cos_c = math.cos(rad)
            sin_c = math.sin(rad)

            # Signed seaward distance: y_sea > 0 in ocean, y_sea <= 0 inland
            y_sea = -self.X * sin_c + self.Y * cos_c - coastline_offset

            # A. Offshore Dean's Profile + Marine Bathymetry Transition (y_sea > 0)
            offshore_mask = y_sea > 0.0
            if np.any(offshore_mask):
                y_off = y_sea[offshore_mask]
                h_dean = beach_slope_coefficient * (y_off ** (2.0 / 3.0))
                z_dean = sea_level - h_dean
                z_deep = sea_level - deep_water_depth

                # Cubic smoothstep blend to abyssal continental shelf
                trans_w = 25.0
                t_blend = np.clip((y_off - shelf_transition_dist) / trans_w, 0.0, 1.0)
                s_blend = 3.0 * t_blend**2 - 2.0 * t_blend**3
                z_offshore = (1.0 - s_blend) * z_dean + s_blend * z_deep

                # Carve offshore seabed (preserving existing lower elevations)
                self.Z[offshore_mask] = np.minimum(self.Z[offshore_mask], z_offshore)

            # B. Inland Supratidal Beach Berm (0 <= d_inland <= beach_width)
            if beach_width > 0.0 and berm_elevation > 0.0:
                inland_dist = -y_sea
                berm_mask = (inland_dist >= 0.0) & (inland_dist <= beach_width)
                if np.any(berm_mask):
                    u_berm = inland_dist[berm_mask] / beach_width
                    s_berm = 3.0 * u_berm**2 - 2.0 * u_berm**3
                    target_berm = sea_level + berm_elevation * s_berm
                    # Gently elevate shoreline depressions to form sandy berm platform
                    self.Z[berm_mask] = np.maximum(self.Z[berm_mask], target_berm)

        # Part 2: Wave-Cut Cliff Notch Synthesis (carved into steep rock near sea level)
        if wave_notch_depth > 0.0 and wave_notch_envelope > 0.0:
            sigma_sq = 2.0 * (wave_notch_envelope ** 2)
            elevation_diff = self.Z - sea_level
            gaussian_notch = wave_notch_depth * np.exp(-(elevation_diff ** 2) / sigma_sq)

            # Intertidal envelope: within +/- 2.5 * envelope
            intertidal_mask = np.abs(elevation_diff) <= 2.5 * wave_notch_envelope
            if np.any(intertidal_mask):
                # Concentrate notch carving on steeper faces (cliffs)
                slopes = self.compute_slope_degrees()
                cliff_weight = np.clip(slopes / 25.0, 0.0, 1.0)
                self.Z[intertidal_mask] -= (gaussian_notch * cliff_weight)[intertidal_mask]

    def apply_geological_features(self, config: MapConfig) -> None:
        """
        High-level orchestration method that reads MapConfig and applies all
        configured tectonic, strata, terracing, and coastal features.
        """
        # 1. Tectonic Fault Scarp
        if getattr(config, "tectonic", None) and config.tectonic.enabled:
            disp = getattr(config.tectonic, "displacement", None)
            if disp is None:
                disp = getattr(config.tectonic, "displacement_throw", 14.0)
            ang = getattr(config.tectonic, "angle", None)
            if ang is None:
                ang = getattr(config.tectonic, "fault_angle_deg", -25.0)
            self.apply_tectonic_scarp(
                fault_angle_deg=ang,
                displacement_throw=disp,
                scarp_sharpness=config.tectonic.scarp_sharpness,
                fault_offset=config.tectonic.fault_offset,
                fault_length=config.tectonic.fault_length,
            )

        # 2. Strata Folding
        if getattr(config, "strata", None) and config.strata.enabled:
            self.apply_strata_folding(
                amplitude=config.strata.amplitude,
                wavelength=config.strata.wavelength,
                strike_angle_deg=config.strata.strike_angle_deg,
                plunge_wavelength=config.strata.plunge_wavelength,
                harmonic_weight=config.strata.harmonic_weight,
                domain_warp_amp=config.strata.domain_warp_amp,
                domain_warp_freq=config.strata.domain_warp_freq,
            )

        # 3. Canyon Terracing
        if getattr(config, "strata", None) and (
            getattr(config.strata, "terrace_enabled", False)
            or getattr(config.strata, "canyon_terracing", False)
        ):
            h = getattr(config.strata, "step_height", None)
            if h is None:
                h = getattr(config.strata, "terrace_height", 6.0)
            k = getattr(config.strata, "steepness", None)
            if k is None:
                k = getattr(config.strata, "terrace_sharpness", 6.0)
            b = getattr(config.strata, "blend", None)
            if b is None:
                b = getattr(config.strata, "terrace_blend", 0.85)
            self.apply_canyon_terraces(
                terrace_height=h,
                terrace_sharpness=k,
                blend=b,
                noise_modulation=getattr(config.strata, "noise_modulation", 0.0),
            )

        # 4. Coastal Profile & Wave-Cut Notches
        if getattr(config, "coastal", None) and config.coastal.enabled:
            self.apply_coastal_profile(
                sea_level=config.coastal.sea_level,
                wave_notch_depth=config.coastal.wave_notch_depth,
                wave_notch_envelope=config.coastal.wave_notch_envelope,
                beach_slope_coefficient=config.coastal.beach_slope_coefficient,
                berm_elevation=config.coastal.berm_elevation,
                beach_width=config.coastal.beach_width,
                shelf_transition_dist=config.coastal.shelf_transition_dist,
                deep_water_depth=config.coastal.deep_water_depth,
                coastline_angle_deg=config.coastal.coastline_angle_deg,
                coastline_offset=config.coastal.coastline_offset,
            )


    def ensure_cavern_clearance(self, config: MapConfig, min_clearance: float = 1.5) -> None:
        """
        Guarantees that the mountain terrain surface encloses the entire 2D elliptical
        karst cavern chamber footprint with at least `min_clearance` rock overburden thickness.
        Prevents subterranean breach where cavern ceiling punches through mountain flanks.
        """
        if not getattr(config, "cavern", None) or not config.cavern.enabled:
            return
        cx, cy, cz = config.cavern.center
        rx = getattr(config.cavern, "radius_x", 0.0)
        ry = getattr(config.cavern, "radius_y", 0.0)
        h = getattr(config.cavern, "height", 0.0)
        if rx <= 0.0 or ry <= 0.0 or h <= 0.0:
            return

        q = ((self.X - cx) / rx) ** 2 + ((self.Y - cy) / ry) ** 2
        in_cavern = q <= 1.0
        if np.any(in_cavern):
            local_roof_z = cz + h * np.sqrt(np.maximum(0.0, 1.0 - q[in_cavern]))
            min_surface_z = local_roof_z + min_clearance
            self.Z[in_cavern] = np.maximum(self.Z[in_cavern], min_surface_z)

    def apply_bilateral_filter(
        self,
        spatial_sigma: float = 2.5,
        range_sigma: float = 1.8,
        radius: int = 3,
        iterations: int = 1,
    ) -> None:
        """
        Applies edge-preserving bilateral surface smoothing across the heightfield.
        Eliminates terracing, stepping artifacts, and micro-ridges on gentle slopes
        while preserving razor-sharp aretes, horn summits, and tectonic fault scarps.
        """
        if iterations <= 0 or spatial_sigma <= 0.0 or range_sigma <= 0.0 or radius <= 0:
            return

        ny, nx = self.res_y, self.res_x
        two_ssq = 2.0 * (spatial_sigma ** 2)
        two_rsq = 2.0 * (range_sigma ** 2)

        # Precompute spatial Gaussian weights for (2r+1) x (2r+1) window
        dy_grid, dx_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        spatial_weights = np.exp(-(dx_grid**2 + dy_grid**2) / two_ssq)

        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append((dy, dx, spatial_weights[dy + radius, dx + radius]))

        y_indices, x_indices = np.indices((ny, nx))

        for _ in range(iterations):
            numerator = np.zeros_like(self.Z)
            denominator = np.zeros_like(self.Z)
            Z_cur = self.Z

            for dy, dx, w_spatial in offsets:
                shifted_y = np.clip(y_indices + dy, 0, ny - 1)
                shifted_x = np.clip(x_indices + dx, 0, nx - 1)
                Z_shifted = Z_cur[shifted_y, shifted_x]

                diff = Z_shifted - Z_cur
                w_range = np.exp(-(diff ** 2) / two_rsq)
                w_total = w_spatial * w_range

                numerator += w_total * Z_shifted
                denominator += w_total

            self.Z = numerator / np.maximum(denominator, 1e-12)

    def apply_laplacian_smoothing(
        self,
        iterations: int = 1,
        lambda_rate: float = 0.20,
        anti_shrink: bool = True,
    ) -> None:
        """
        Applies discrete 8-neighbor Laplacian surface relaxation.
        When anti_shrink=True, uses Taubin two-step filtering (lambda > 0, mu < -lambda)
        to eliminate high-frequency grid noise and single-pixel erosion spikes without volume loss.
        """
        if iterations <= 0 or lambda_rate <= 0.0:
            return

        ny, nx = self.res_y, self.res_x
        y_idx, x_idx = np.indices((ny, nx))

        # 8 neighbor offsets with Euclidean distance weighting
        nbr_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        nbr_weights = [1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678]
        w_sum = sum(nbr_weights)

        # Taubin parameters: lambda > 0, mu < 0 (balanced smoothing and anti-shrinkage)
        mu_rate = -lambda_rate * 0.50 if anti_shrink else 0.0
        steps = [lambda_rate, mu_rate] if anti_shrink else [lambda_rate]

        for _ in range(iterations):
            for factor in steps:
                sum_nbrs = np.zeros_like(self.Z)
                for (dy, dx), w in zip(nbr_offsets, nbr_weights):
                    sy = np.clip(y_idx + dy, 0, ny - 1)
                    sx = np.clip(x_idx + dx, 0, nx - 1)
                    sum_nbrs += w * self.Z[sy, sx]

                laplacian = (sum_nbrs / w_sum) - self.Z
                self.Z += factor * laplacian

    def add_ridged_multifractal(
        self,
        seed: int = 42,
        octaves: int = 4,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        base_frequency: float = 0.025,
        amplitude: float = 6.0,
    ) -> None:
        """
        Synthesizes Ridged Multi-Fractal (RMF) terrain noise using C2 Perlin gradient noise.
        Produces crisp, jagged alpine rock ribs, knife-edge aretes, and frost-shattered peaks.
        """
        total = np.zeros_like(self.Z)
        cur_amp = amplitude
        cur_freq = base_frequency
        weight = 1.0

        for i in range(octaves):
            n = _perlin_2d(self.X, self.Y, cur_freq, seed=seed + i * 101)
            # Inverted absolute noise gives ridge crests
            signal = 1.0 - np.abs(n)
            signal = signal * signal  # Sharpen ridge
            signal *= weight

            weight = np.clip(signal * 2.0, 0.0, 1.0)
            total += signal * cur_amp

            cur_freq *= lacunarity
            cur_amp *= gain

        self.Z += total


