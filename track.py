"""
Track module for the AI Driving Simulation.

Defines a smooth race track with gently curved corners, generated
parametrically from a center line with outward normals.
"""

import math
import pygame


class Track:
    """A smooth race track defined by wall segments, checkpoints, and a spawn point."""

    # Colors - Road & Track
    ROAD_COLOR = (60, 60, 60)
    ROAD_EDGE_COLOR = (200, 200, 200)  # Bright white edge lines
    ROAD_EDGE_WIDTH = 3

    # Colors - Walls & Curbs
    WALL_COLOR = (40, 40, 40)  # Dark wall line
    WALL_WIDTH = 2
    CURB_RED = (210, 25, 25)
    CURB_WHITE = (235, 235, 235)
    CURB_WIDTH = 4
    CURB_SEGMENTS_PER_BAND = 4  # Number of segments per curb color band

    # Colors - Grass
    GRASS_COLOR = (30, 85, 30)
    GRASS_STRIPE_LIGHT = (35, 92, 35)
    GRASS_STRIPE_DARK = (26, 75, 26)
    GRASS_STRIPE_INTERVAL = 36  # Pixels between grass stripe centers

    # Colors - Other
    CHECKPOINT_COLOR = (255, 255, 0)
    CHECKPOINT_PASSED_COLOR = (0, 200, 0)
    SPAWN_COLOR = (0, 255, 100)

    def __init__(self):
        """Initialize the track with smooth walls, checkpoints, and spawn point."""
        self.width = 1200
        self.height = 800

        # ---- Track geometry parameters ----
        # A single continuous closed-loop circuit with smooth, wide corners.
        # Defined via waypoints and smoothed.
        half_width = 40  # Road width (50px total) - safe for tight curves

        # ---- Waypoints defining the centre line (clockwise) ----
        # Minimal perimeter circuit. Uses very few waypoints with
        # minimal smoothing to guarantee zero self-intersection.
        waypoints = [
            (100, 740),   # Start/bottom-left
            (600, 740),   # Bottom straight
            (1050, 735),  # End of main straight

            (1100, 690),  # Right turn entry
            (1115, 600),  # Right side up
            (1110, 500),  # Right side
            (1080, 400),  # Right side top

            (1000, 335),  # Top-right
            (800, 320),   # Top straight
            (500, 325),   # Top straight
            (250, 335),   # Top-left

            (140, 400),   # Left side down
            (120, 550),   # Left side
            (110, 680),   # Left side bottom

            (105, 740),   # Final into straight
        ]

        # ---- Minimal smoothing to prevent clustering ----
        self._center_points = Track._smooth_closed_path(waypoints,
                                                        subdivisions=1,
                                                        passes=2)
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

        # ---- Checkpoints evenly spaced around the track ----
        num_cp = max(4, min(8, n // 10))
        cp_indices = [i * n // num_cp for i in range(num_cp)]
        self.checkpoints = []
        for idx in cp_indices:
            self.checkpoints.append((
                (inner_boundary[idx][0], inner_boundary[idx][1]),
                (outer_boundary[idx][0], outer_boundary[idx][1]),
            ))
        self.num_checkpoints = len(self.checkpoints)
        self.total_checkpoints = self.num_checkpoints

        # ---- Spawn point — first smoothed centre point, facing right ----
        # Use the smoothed centre line so the car always spawns between the walls
        self.spawn_x, self.spawn_y = self._center_points[0]
        tx, ty = self._center_points[1 % n]
        self.spawn_angle = math.atan2(ty - self.spawn_y, tx - self.spawn_x)

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

        # ---- 4. Spawn at the first smoothed centre point, facing the second ----
        track.spawn_x, track.spawn_y = track._center_points[0]
        tx, ty = track._center_points[1] if len(track._center_points) > 1 else track._center_points[0]
        track.spawn_angle = math.atan2(ty - track.spawn_y, tx - track.spawn_x)

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

        n = len(self._outer_boundary)

        # ============================================================
        # 1. Grass base with texture stripes
        # ============================================================
        surface.fill(self.GRASS_COLOR)

        # Draw subtle diagonal grass stripes (mown-pattern look)
        # These only show through on the grass since the road is drawn on top
        for offset in range(-self.height * 2, self.width + self.height * 2, self.GRASS_STRIPE_INTERVAL):
            stripe_idx = (offset // self.GRASS_STRIPE_INTERVAL) % 2
            color = self.GRASS_STRIPE_LIGHT if stripe_idx == 0 else self.GRASS_STRIPE_DARK
            start_x = ox + offset
            start_y = oy
            end_x = ox + offset - self.height
            end_y = oy + self.height
            pygame.draw.line(surface, color,
                             (start_x, start_y),
                             (end_x, end_y), 4)

        # ============================================================
        # 2. Road surface (individual quads - robust against overlap)
        # ============================================================
        for i in range(n):
            j = (i + 1) % n
            quad = [
                (self._outer_boundary[i][0] + ox, self._outer_boundary[i][1] + oy),
                (self._outer_boundary[j][0] + ox, self._outer_boundary[j][1] + oy),
                (self._inner_boundary[j][0] + ox, self._inner_boundary[j][1] + oy),
                (self._inner_boundary[i][0] + ox, self._inner_boundary[i][1] + oy),
            ]
            pygame.draw.polygon(surface, self.ROAD_COLOR, quad)

        # ============================================================
        # 3. Curbs (alternating red/white strips along both edges)
        # ============================================================
        for i in range(n):
            j = (i + 1) % n
            is_red = (i // self.CURB_SEGMENTS_PER_BAND) % 2 == 0
            curb_color = self.CURB_RED if is_red else self.CURB_WHITE

            # Outer curb
            p1_o = self._outer_boundary[i]
            p2_o = self._outer_boundary[j]
            pygame.draw.line(surface, curb_color,
                             (p1_o[0] + ox, p1_o[1] + oy),
                             (p2_o[0] + ox, p2_o[1] + oy),
                             self.CURB_WIDTH)

            # Inner curb
            p1_i = self._inner_boundary[i]
            p2_i = self._inner_boundary[j]
            pygame.draw.line(surface, curb_color,
                             (p1_i[0] + ox, p1_i[1] + oy),
                             (p2_i[0] + ox, p2_i[1] + oy),
                             self.CURB_WIDTH)

        # ============================================================
        # 4. White edge lines (on top of curbs for crisp road boundary)
        # ============================================================
        for i in range(n):
            j = (i + 1) % n
            # Outer edge
            p1_o = self._outer_boundary[i]
            p2_o = self._outer_boundary[j]
            pygame.draw.line(surface, self.ROAD_EDGE_COLOR,
                             (p1_o[0] + ox, p1_o[1] + oy),
                             (p2_o[0] + ox, p2_o[1] + oy),
                             self.ROAD_EDGE_WIDTH)
            # Inner edge
            p1_i = self._inner_boundary[i]
            p2_i = self._inner_boundary[j]
            pygame.draw.line(surface, self.ROAD_EDGE_COLOR,
                             (p1_i[0] + ox, p1_i[1] + oy),
                             (p2_i[0] + ox, p2_i[1] + oy),
                             self.ROAD_EDGE_WIDTH)

        # ============================================================
        # 5. Dashed centre line (yellow-white)
        # ============================================================
        for i in range(len(self._center_points)):
            p1 = self._center_points[i]
            p2 = self._center_points[(i + 1) % len(self._center_points)]
            self._draw_dashed_line(
                surface, (220, 220, 80),
                (p1[0] + ox, p1[1] + oy),
                (p2[0] + ox, p2[1] + oy),
                3, 18, 12,
            )

        # ============================================================
        # 6. Wall lines (thin dark boundary)
        # ============================================================
        for wall in self.walls:
            p1, p2 = wall
            pygame.draw.line(surface, self.WALL_COLOR,
                             (p1[0] + ox, p1[1] + oy),
                             (p2[0] + ox, p2[1] + oy),
                             self.WALL_WIDTH)

        # ============================================================
        # 7. Checkpoints
        # ============================================================
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

        # ============================================================
        # 8. Spawn indicator
        # ============================================================
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

        # ============================================================
        # 9. Checkpoint labels
        # ============================================================
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
