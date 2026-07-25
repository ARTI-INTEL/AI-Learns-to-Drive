"""
Car module for the AI Driving Simulation.

Defines the Car class with physics, sensor rays (raycasting), collision detection,
and rendering. The car can be controlled manually (keyboard) or by a neural network.
"""

import math
import pygame


def line_intersection(p1, p2, p3, p4):
    """
    Check if line segments p1-p2 and p3-p4 intersect.

    Args:
        p1, p2: Endpoints of first line segment
        p3, p4: Endpoints of second line segment

    Returns:
        Tuple (ix, iy, dist) if intersection found, None otherwise.
        ix, iy: Intersection point
        dist: Distance from p1 to intersection point
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        dist = math.sqrt((ix - x1) ** 2 + (iy - y1) ** 2)
        return ix, iy, dist

    return None


class Car:
    """A car with position, velocity, steering, and sensor rays."""

    # Car dimensions
    WIDTH = 20
    HEIGHT = 12
    COLLISION_RADIUS = 14  # Circle approximation for collision

    # Physics constants
    ACCELERATION = 0.15
    BRAKE_FORCE = 0.08
    FRICTION = 0.98
    MAX_SPEED = 4.0
    MAX_REVERSE_SPEED = 1.5
    TURN_SPEED = 0.045  # radians per frame at max speed
    TURN_SPEED_FACTOR = 0.6  # reduces turning at high speed

    # Sensor configuration
    NUM_SENSORS = 7
    SENSOR_ANGLES = [-60, -35, -15, 0, 15, 35, 60]  # degrees relative to car direction
    MAX_SENSOR_DISTANCE = 400

    # Colors
    CAR_COLOR = (50, 150, 255)
    CAR_CRASHED_COLOR = (200, 50, 50)
    SENSOR_GREEN = (0, 255, 0)
    SENSOR_RED = (255, 0, 0)
    SENSOR_HIT_COLOR = (255, 200, 0)

    def __init__(self, x, y, angle=0.0, genome_id=None):
        """
        Initialize the car.

        Args:
            x: Start x position
            y: Start y position
            angle: Starting direction in radians (0 = right)
            genome_id: Optional identifier for NEAT genome
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.steering = 0.0  # -1 to 1

        self.genome_id = genome_id

        # State tracking
        self.alive = True
        self.crashed = False
        self.finished = False
        self.collision_wall = None

        # Fitness tracking
        self.total_distance = 0.0
        self.last_x = x
        self.last_y = y
        self.max_speed_reached = 0.0
        self.time_stuck = 0.0
        self.checkpoints_passed = set()
        self.lap = 0
        self.fitness = 0.0

        # Sensor readings (populated each frame)
        self.sensor_readings = [1.0] * self.NUM_SENSORS
        self.sensor_hit_points = [None] * self.NUM_SENSORS
        self.sensor_hit_wall = [False] * self.NUM_SENSORS

        # Checkpoint tracking
        self.total_checkpoints = 4  # Updated by simulation/track
        self.last_checkpoint_reward = -1

        # Circle detection tracking
        self.total_angle_change = 0.0
        self.prev_angle = angle

    def reset(self, x, y, angle):
        """Reset the car to a given position."""
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.steering = 0.0
        self.alive = True
        self.crashed = False
        self.finished = False
        self.collision_wall = None
        self.total_distance = 0.0
        self.last_x = x
        self.last_y = y
        self.max_speed_reached = 0.0
        self.time_stuck = 0.0
        self.checkpoints_passed = set()
        self.lap = 0
        self.fitness = 0.0
        self.sensor_readings = [1.0] * self.NUM_SENSORS
        self.sensor_hit_points = [None] * self.NUM_SENSORS
        self.sensor_hit_wall = [False] * self.NUM_SENSORS
        self.last_checkpoint_reward = -1
        self.total_angle_change = 0.0
        self.prev_angle = angle

    def update(self, steering, throttle, walls, dt=1.0):
        """
        Update the car's physics for one frame.

        Args:
            steering: Steering input (-1 to 1, where -1=left, 1=right)
            throttle: Throttle input (-1 to 1, where negative=brake/reverse, positive=accelerate)
            walls: List of wall segments to check against
            dt: Delta time factor
        """
        if not self.alive:
            return

        self.steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))

        # ---- Physics ----
        # Apply throttle
        if throttle > 0:
            self.speed += throttle * self.ACCELERATION * dt
        elif throttle < 0:
            # Brake or reverse
            if self.speed > 0:
                self.speed += throttle * self.BRAKE_FORCE * dt  # negative throttle slows down
            else:
                self.speed += throttle * self.ACCELERATION * dt * 0.5  # reverse is slower

        # Apply friction
        self.speed *= self.FRICTION

        # Clamp speed
        if self.speed > self.MAX_SPEED:
            self.speed = self.MAX_SPEED
        elif self.speed < -self.MAX_REVERSE_SPEED:
            self.speed = -self.MAX_REVERSE_SPEED

        # Track max speed
        if abs(self.speed) > self.max_speed_reached:
            self.max_speed_reached = abs(self.speed)

        # Apply steering (turning is speed-dependent)
        turn_factor = self.TURN_SPEED_FACTOR + (1 - self.TURN_SPEED_FACTOR) * (1 - abs(self.speed) / self.MAX_SPEED)
        new_angle = self.angle + self.steering * self.TURN_SPEED * max(0.3, turn_factor) * dt

        # Track total angle change (for circle detection)
        angle_diff = abs(new_angle - self.prev_angle)
        # Handle angle wrap-around
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        self.total_angle_change += angle_diff
        self.prev_angle = new_angle
        self.angle = new_angle

        # Store previous position
        prev_x, prev_y = self.x, self.y

        # Move the car
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

        # Track distance moved
        dx = self.x - self.last_x
        dy = self.y - self.last_y
        moved = math.sqrt(dx * dx + dy * dy)
        self.total_distance += moved
        self.last_x = self.x
        self.last_y = self.y

        # Track if car is stuck (barely moving)
        if moved < 0.5:
            self.time_stuck += 1
        else:
            self.time_stuck = 0

        # ---- Collision detection ----
        if self._check_collision(walls):
            self.alive = False
            self.crashed = True
            # Move back to previous position
            self.x, self.y = prev_x, prev_y

        # ---- Cast sensor rays ----
        self._cast_sensors(walls)

    def _check_collision(self, walls):
        """Check if the car is colliding with any wall."""
        for wall in walls:
            p1, p2 = wall
            # Distance from car center to line segment
            dist = self._point_to_segment_distance((self.x, self.y), p1, p2)
            if dist < self.COLLISION_RADIUS + 2:  # 2px buffer
                self.collision_wall = wall
                return True
        return False

    def _point_to_segment_distance(self, p, a, b):
        """Calculate the minimum distance from point p to line segment a-b."""
        px, py = p
        ax, ay = a
        bx, by = b

        # Vector from a to b
        abx = bx - ax
        aby = by - ay

        # Vector from a to p
        apx = px - ax
        apy = py - ay

        # Project p onto ab
        ab_sq = abx * abx + aby * aby
        if ab_sq == 0:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

        t = (apx * abx + apy * aby) / ab_sq
        t = max(0.0, min(1.0, t))

        # Closest point on segment
        cx = ax + t * abx
        cy = ay + t * aby

        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    def _cast_sensors(self, walls):
        """Cast sensor rays from the car and detect wall intersections."""
        for i, sensor_angle_deg in enumerate(self.SENSOR_ANGLES):
            sensor_angle = self.angle + math.radians(sensor_angle_deg)

            # Sensor ray endpoint
            end_x = self.x + math.cos(sensor_angle) * self.MAX_SENSOR_DISTANCE
            end_y = self.y + math.sin(sensor_angle) * self.MAX_SENSOR_DISTANCE

            closest_dist = self.MAX_SENSOR_DISTANCE
            closest_hit = None
            hit_wall = False

            # Check against all walls
            for wall in walls:
                result = line_intersection(
                    (self.x, self.y), (end_x, end_y),
                    wall[0], wall[1]
                )
                if result is not None:
                    _, _, dist = result
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hit = (result[0], result[1])
                        hit_wall = True

            # Normalize sensor reading to 0-1 (1 = no wall, 0 = wall right at car)
            self.sensor_readings[i] = closest_dist / self.MAX_SENSOR_DISTANCE
            self.sensor_hit_points[i] = closest_hit
            self.sensor_hit_wall[i] = hit_wall

    def get_inputs(self):
        """
        Get neural network inputs from sensor readings and car state.

        Returns:
            List of 8 float values: 7 sensors (normalized 0-1) + speed (normalized 0-1)
        """
        inputs = list(self.sensor_readings)  # 7 sensors
        inputs.append(abs(self.speed) / self.MAX_SPEED)  # normalized speed
        return inputs

    def draw(self, surface, camera_offset=(0, 0), show_sensors=True):
        """
        Draw the car and its sensors on the given surface.

        Args:
            surface: Pygame surface to draw on
            camera_offset: (ox, oy) camera offset
            show_sensors: Whether to draw sensor rays
        """
        ox, oy = camera_offset
        cx = int(self.x + ox)
        cy = int(self.y + oy)

        # ---- Draw sensor rays ----
        if show_sensors and self.alive:
            for i in range(self.NUM_SENSORS):
                sensor_angle = self.angle + math.radians(self.SENSOR_ANGLES[i])
                reading = self.sensor_readings[i]
                hit_point = self.sensor_hit_points[i]
                hit_wall = self.sensor_hit_wall[i]

                # Ray endpoint (based on actual hit distance)
                ray_length = reading * self.MAX_SENSOR_DISTANCE
                end_x = int(self.x + math.cos(sensor_angle) * ray_length + ox)
                end_y = int(self.y + math.sin(sensor_angle) * ray_length + oy)

                # Color: green if clear, yellow if hit, red if close hit
                if hit_wall:
                    if reading < 0.2:
                        color = self.SENSOR_RED
                    elif reading < 0.5:
                        color = self.SENSOR_HIT_COLOR
                    else:
                        color = self.SENSOR_GREEN
                else:
                    color = self.SENSOR_GREEN

                # Draw ray with alpha-like effect using line width
                width = 1 if not hit_wall else 2
                pygame.draw.line(surface, color, (cx, cy), (end_x, end_y), width)

                # Draw hit point
                if hit_point is not None:
                    hx = int(hit_point[0] + ox)
                    hy = int(hit_point[1] + oy)
                    pygame.draw.circle(surface, (255, 100, 0), (hx, hy), 3)

        # ---- Draw car body ----
        # Create car surface for rotation
        car_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        car_surf.set_colorkey((0, 0, 0))

        # Car body color
        if not self.alive:
            body_color = self.CAR_CRASHED_COLOR
        else:
            body_color = self.CAR_COLOR

        # Main body (rounded rectangle)
        body_rect = pygame.Rect(1, 2, self.WIDTH - 2, self.HEIGHT - 4)
        pygame.draw.ellipse(car_surf, body_color, body_rect)
        pygame.draw.ellipse(car_surf, (200, 200, 255), body_rect, 1)

        # Windshield
        windshield_rect = pygame.Rect(self.WIDTH - 12, 3, 8, self.HEIGHT - 6)
        pygame.draw.ellipse(car_surf, (150, 200, 255), windshield_rect)
        pygame.draw.ellipse(car_surf, (100, 150, 200), windshield_rect, 1)

        # Headlights
        pygame.draw.circle(car_surf, (255, 255, 200), (self.WIDTH - 3, 4), 2)
        pygame.draw.circle(car_surf, (255, 255, 200), (self.WIDTH - 3, self.HEIGHT - 4), 2)

        # Tail lights
        pygame.draw.circle(car_surf, (255, 50, 50), (3, 4), 2)
        pygame.draw.circle(car_surf, (255, 50, 50), (3, self.HEIGHT - 4), 2)

        # Rotate car surface
        angle_deg = -math.degrees(self.angle)  # pygame rotation is clockwise, our angle is CCW
        rotated_surf = pygame.transform.rotate(car_surf, angle_deg)
        rotated_rect = rotated_surf.get_rect(center=(cx, cy))
        surface.blit(rotated_surf, rotated_rect)

        # ---- Draw genome ID label ----
        if self.genome_id is not None:
            font = pygame.font.SysFont('Arial', 11)
            label = font.render(f'#{self.genome_id}', True, (255, 255, 255))
            surface.blit(label, (cx - 12, cy + 18))

    def get_checkpoint_progress(self, checkpoints):
        """
        Check if the car has crossed any checkpoint.
        Returns the index of the crossed checkpoint, or -1.
        """
        for i, cp in enumerate(checkpoints):
            if i in self.checkpoints_passed:
                continue
            if self._line_intersects_car_path(cp):
                self.checkpoints_passed.add(i)
                return i
        return -1

    def _line_intersects_car_path(self, checkpoint_line):
        """
        Check if the car's path (simplified as the line from last position to current)
        intersects a checkpoint line.

        Uses a simple point-in-front-of-checkpoint test instead, which is more robust.
        """
        cp1, cp2 = checkpoint_line

        # Calculate checkpoint normal (pointing in the direction the car should cross)
        # For a vertical checkpoint (cp1-top, cp2-bottom), normal points right
        # For a horizontal checkpoint, normal points down
        cx = (cp1[0] + cp2[0]) / 2
        cy = (cp1[1] + cp2[1]) / 2

        # Vector along the checkpoint
        vx = cp2[0] - cp1[0]
        vy = cp2[1] - cp1[1]

        # Perpendicular vector (normal)
        nx = -vy
        ny = vx

        # Normalize
        n_len = math.sqrt(nx * nx + ny * ny)
        if n_len > 0:
            nx /= n_len
            ny /= n_len

        # Check if the car crossed by seeing if it's on the same side of the line
        # as the average of previous positions
        # Simple approach: check intersection of car path with checkpoint
        return line_intersection(
            (self.last_x, self.last_y), (self.x, self.y),
            cp1, cp2
        ) is not None
