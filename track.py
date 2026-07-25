"""
Track module for the AI Driving Simulation.

Defines a smooth race track with gently curved corners, generated
parametrically from a center line with outward normals.
"""

import math
import pygame


class Track:
    """A smooth race track defined by wall segments, checkpoints, and a spawn point."""

    # Colors
    ROAD_COLOR = (60, 60, 60)
    ROAD_EDGE_COLOR = (100, 100, 100)
    WALL_COLOR = (200, 50, 50)
    WALL_WIDTH = 5
    CHECKPOINT_COLOR = (255, 255, 0)
    CHECKPOINT_PASSED_COLOR = (0, 200, 0)
    SPAWN_COLOR = (0, 255, 100)
    GRASS_COLOR = (30, 80, 30)

    def __init__(self):
        """Initialize the track with smooth walls, checkpoints, and spawn point."""
        self.width = 1200
        self.height = 800

        # ---- Track geometry parameters ----
        # The track is a smooth oval with two long straights and two semi-circular curves.
        # Centre of the oval.
        oval_cx, oval_cy = 600, 400
        # How far the straights extend left/right from centre.
        straight_half = 200          # each straight is 400 px long
        # Radius of the semicircular curves (at the centre line).
        curve_r = 220
        # Half-width of the road (from centre line to one wall).
        half_width = 40

        # ---- Generate centre-line points (clockwise) ----
        self._center_points = []   # list of (x, y)

        # ① Top straight  — left ➔ right (exclude final point; the curve covers it)
        x0, x1 = oval_cx - straight_half, oval_cx + straight_half
        y_top = oval_cy - curve_r
        steps = 10
        for i in range(steps - 1):
            t = i / (steps - 1) if steps > 1 else 0
            self._center_points.append((x0 + t * (x1 - x0), y_top))

        # ② Right curve — semicircle from -90° to +90°
        right_cx = oval_cx + straight_half
        for i in range(21):       # 0…20 → 20 segments
            angle = -math.pi / 2 + i * math.pi / 20
            self._center_points.append((
                right_cx + curve_r * math.cos(angle),
                oval_cy + curve_r * math.sin(angle),
            ))

        # ③ Bottom straight — right ➔ left (exclude final point; the curve covers it)
        x0, x1 = oval_cx + straight_half, oval_cx - straight_half
        y_bot = oval_cy + curve_r
        steps = 10
        for i in range(steps - 1):
            t = i / (steps - 1) if steps > 1 else 0
            self._center_points.append((x0 + t * (x1 - x0), y_bot))

        # ④ Left curve — semicircle from +90° to +270°
        left_cx = oval_cx - straight_half
        for i in range(21):
            angle = math.pi / 2 + i * math.pi / 20
            self._center_points.append((
                left_cx + curve_r * math.cos(angle),
                oval_cy + curve_r * math.sin(angle),
            ))

        n = len(self._center_points)

        # ---- Compute outward normals and build wall segments ----
        def _normal(i):
            """Return outward-facing unit normal at centre point *i*."""
            prev = self._center_points[(i - 1) % n]
            curr = self._center_points[i]
            nxt = self._center_points[(i + 1) % n]
            dx = nxt[0] - prev[0]
            dy = nxt[1] - prev[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                return (0.0, 0.0)
            dx /= length
            dy /= length
            # Perpendicular: (dy, -dx)  — rotates tangent 90° CW.
            # For our clockwise track this points outward.
            return (dy, -dx)

        outer_boundary = []
        inner_boundary = []
        for i, (cx, cy) in enumerate(self._center_points):
            nx, ny = _normal(i)
            outer_boundary.append((cx + nx * half_width, cy + ny * half_width))
            inner_boundary.append((cx - nx * half_width, cy - ny * half_width))

        # Turn boundary points into wall segments
        self.walls = []
        for i in range(n):
            j = (i + 1) % n
            self.walls.append((outer_boundary[i], outer_boundary[j]))
            self.walls.append((inner_boundary[i], inner_boundary[j]))

        # Store the smoothed boundary for road drawing
        self._outer_boundary = outer_boundary
        self._inner_boundary = inner_boundary

        # ---- Checkpoints (span from inner → outer wall) ----
        # Pick indices along the centre line where checkpoints sit.
        # Top-straight centre, right-curve centre, bottom-straight centre,
        # left-curve centre.
        cp_indices = [
            len(self._center_points) // 8,          # top straight middle
            len(self._center_points) * 3 // 8,      # right curve middle
            len(self._center_points) * 5 // 8,      # bottom straight middle
            len(self._center_points) * 7 // 8,      # left curve middle
        ]
        self.checkpoints = []
        for idx in cp_indices:
            self.checkpoints.append((
                (inner_boundary[idx][0], inner_boundary[idx][1]),
                (outer_boundary[idx][0], outer_boundary[idx][1]),
            ))

        self.num_checkpoints = len(self.checkpoints)
        self.total_checkpoints = self.num_checkpoints

        # ---- Spawn point — middle of the top straight, facing right ----
        self.spawn_x = oval_cx
        self.spawn_y = y_top
        self.spawn_angle = 0.0

        # ---- Centre line for dashed road markings ----
        self.center_line = self._center_points[:]

    # ------------------------------------------------------------------
    #  Build from user-drawn waypoints
    # ------------------------------------------------------------------

    @classmethod
    def from_waypoints(cls, waypoints, half_width=50):
        """
        Create a Track from a list of user-drawn (x, y) waypoints.

        The waypoints define the centre line of the track (an open path
        that will be closed into a loop).  Points are smoothed and
        subdivided to produce a clean racing line.

        Args:
            waypoints: List of (x, y) tuples.
            half_width: Half the road width in pixels.

        Returns:
            A fully-built Track instance.
        """
        if len(waypoints) < 3:
            raise ValueError("Need at least 3 waypoints")

        # Create a raw instance without running __init__
        track = cls.__new__(cls)
        track.width = 1200
        track.height = 800

        # ---- 1. Smooth the waypoints into a clean centre line ----
        smoothed = cls._smooth_closed_path(waypoints,
                                           subdivisions=3,
                                           passes=4)
        track._center_points = smoothed
        n = len(smoothed)

        # ---- 2. Compute outward normals and build wall boundaries ----
        def _normal(i):
            prev = smoothed[(i - 1) % n]
            nxt = smoothed[(i + 1) % n]
            dx = nxt[0] - prev[0]
            dy = nxt[1] - prev[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                return (0.0, 0.0)
            dx /= length
            dy /= length
            # Perpendicular (dy, -dx) — rotates tangent 90° CW
            # For a clockwise track this points outward.
            return (dy, -dx)

        outer_boundary = []
        inner_boundary = []
        for i, (cx, cy) in enumerate(smoothed):
            nx, ny = _normal(i)
            outer_boundary.append((cx + nx * half_width, cy + ny * half_width))
            inner_boundary.append((cx - nx * half_width, cy - ny * half_width))

        track._outer_boundary = outer_boundary
        track._inner_boundary = inner_boundary

        # Turn boundary points into wall segments
        track.walls = []
        for i in range(n):
            j = (i + 1) % n
            track.walls.append((outer_boundary[i], outer_boundary[j]))
            track.walls.append((inner_boundary[i], inner_boundary[j]))

        # ---- 3. Checkpoints evenly spaced around the track ----
        num_cp = max(4, min(8, n // 8))  # 4–8 checkpoints
        indices = [i * n // num_cp for i in range(num_cp)]
        track.checkpoints = []
        for idx in indices:
            track.checkpoints.append((
                (inner_boundary[idx][0], inner_boundary[idx][1]),
                (outer_boundary[idx][0], outer_boundary[idx][1]),
            ))
        track.num_checkpoints = len(track.checkpoints)
        track.total_checkpoints = track.num_checkpoints

        # ---- 4. Spawn at the first waypoint, facing the second ----
        sx, sy = waypoints[0]
        tx, ty = waypoints[1] if len(waypoints) > 1 else waypoints[0]
        track.spawn_x = sx
        track.spawn_y = sy
        track.spawn_angle = math.atan2(ty - sy, tx - sx)

        # ---- 5. Centre line for road markings ----
        track.center_line = smoothed[:]

        return track

    @staticmethod
    def _smooth_closed_path(points, subdivisions=3, passes=4):
        """
        Take a list of waypoints and produce a smooth closed curve
        by subdividing segments and applying a moving-average filter.
        """
        if len(points) < 3:
            return list(points)

        pts = list(points)  # copy

        # Subdivide
        for _ in range(subdivisions):
            new_pts = []
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                new_pts.append(p1)
                new_pts.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
            pts = new_pts

        # Smoothing passes (moving average)
        for _ in range(passes):
            new_pts = []
            for i in range(len(pts)):
                prev = pts[(i - 1) % len(pts)]
                cur = pts[i]
                nxt = pts[(i + 1) % len(pts)]
                new_pts.append((
                    0.5 * cur[0] + 0.25 * prev[0] + 0.25 * nxt[0],
                    0.5 * cur[1] + 0.25 * prev[1] + 0.25 * nxt[1],
                ))
            pts = new_pts

        return pts

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def get_spawn(self):
        """Return the spawn position and angle."""
        return self.spawn_x, self.spawn_y, self.spawn_angle

    # ------------------------------------------------------------------
    #  Drawing
    # ------------------------------------------------------------------

    def draw(self, surface, camera_offset=(0, 0), passed_checkpoints=None):
        """Draw the track on the given surface."""
        ox, oy = camera_offset
        if passed_checkpoints is None:
            passed_checkpoints = set()

        # Grass
        surface.fill(self.GRASS_COLOR)

        # --- Road surface (filled polygon) ---
        road_poly = self._outer_boundary + self._inner_boundary[::-1]
        # Convert to screen coordinates
        road_poly_screen = [(x + ox, y + oy) for (x, y) in road_poly]
        pygame.draw.polygon(surface, self.ROAD_COLOR, road_poly_screen)

        # --- Road edge markings (thin lighter lines) ---
        # Outer edge
        for i in range(len(self._outer_boundary)):
            p1 = self._outer_boundary[i]
            p2 = self._outer_boundary[(i + 1) % len(self._outer_boundary)]
            pygame.draw.line(surface, self.ROAD_EDGE_COLOR,
                             (p1[0] + ox, p1[1] + oy),
                             (p2[0] + ox, p2[1] + oy), 2)
        # Inner edge
        for i in range(len(self._inner_boundary)):
            p1 = self._inner_boundary[i]
            p2 = self._inner_boundary[(i + 1) % len(self._inner_boundary)]
            pygame.draw.line(surface, self.ROAD_EDGE_COLOR,
                             (p1[0] + ox, p1[1] + oy),
                             (p2[0] + ox, p2[1] + oy), 2)

        # --- Dashed centre line ---
        for i in range(len(self._center_points)):
            p1 = self._center_points[i]
            p2 = self._center_points[(i + 1) % len(self._center_points)]
            self._draw_dashed_line(
                surface, (200, 200, 100),
                (p1[0] + ox, p1[1] + oy),
                (p2[0] + ox, p2[1] + oy),
                3, 15, 10,
            )

        # --- Walls (red) ---
        for wall in self.walls:
            p1, p2 = wall
            pygame.draw.line(surface, self.WALL_COLOR,
                             (p1[0] + ox, p1[1] + oy),
                             (p2[0] + ox, p2[1] + oy),
                             self.WALL_WIDTH)

        # --- Checkpoints ---
        for i, cp in enumerate(self.checkpoints):
            p1, p2 = cp
            if i in passed_checkpoints:
                color = self.CHECKPOINT_PASSED_COLOR
                alpha = 80
            else:
                color = self.CHECKPOINT_COLOR
                alpha = 120
            cp_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(cp_surf, (*color, alpha),
                             (p1[0] + ox, p1[1] + oy),
                             (p2[0] + ox, p2[1] + oy), 4)
            surface.blit(cp_surf, (0, 0))

        # --- Spawn indicator ---
        pygame.draw.circle(
            surface, self.SPAWN_COLOR,
            (int(self.spawn_x + ox), int(self.spawn_y + oy)), 8, 2,
        )
        arrow_len = 20
        end_x = self.spawn_x + math.cos(self.spawn_angle) * arrow_len
        end_y = self.spawn_y + math.sin(self.spawn_angle) * arrow_len
        pygame.draw.line(
            surface, self.SPAWN_COLOR,
            (int(self.spawn_x + ox), int(self.spawn_y + oy)),
            (int(end_x + ox), int(end_y + oy)), 3,
        )

        # --- Checkpoint labels ---
        font = pygame.font.SysFont('Arial', 16)
        for i, cp in enumerate(self.checkpoints):
            p1, _ = cp
            label = font.render(
                f'CP{i + 1} ✓' if i in passed_checkpoints else f'CP{i + 1}',
                True,
                (0, 200, 0) if i in passed_checkpoints else (200, 200, 0),
            )
            surface.blit(label, (int(p1[0] + ox - 10), int(p1[1] + oy - 20)))

    # ------------------------------------------------------------------
    #  Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_dashed_line(surf, color, start_pos, end_pos,
                          width=1, dash_len=15, gap_len=10):
        """Draw a dashed line on a surface."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx, dy = x2 - x1, y2 - y1
        total = math.hypot(dx, dy)
        if total < 1:
            return
        dx /= total
        dy /= total
        cur = 0.0
        while cur < total:
            end = min(cur + dash_len, total)
            pygame.draw.line(surf, color,
                             (x1 + dx * cur, y1 + dy * cur),
                             (x1 + dx * end, y1 + dy * end), width)
            cur += dash_len + gap_len
