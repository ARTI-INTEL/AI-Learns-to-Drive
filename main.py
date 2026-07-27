"""
AI Learns to Drive - Main Entry Point

A reinforcement learning project where an AI learns to drive a car
around a track using sensor rays and a neural network evolved with NEAT.

Controls:
    - Arrow keys / WASD: Manual driving
    - Space: Pause/Resume
    - R: Reset car (manual mode)
    - M: Toggle manual/AI mode
    - T: Toggle training mode
    - [ / ]: Speed up / Slow down simulation
    - F: Toggle fast-forward
    - H: Toggle help overlay
    - Esc: Quit

Modes:
    1. Manual: Drive with keyboard
    2. AI Demo: Watch a pre-trained or best current AI drive
    3. Training: Watch NEAT train in real-time
    4. Headless Training: Train without rendering (fastest)
"""

import sys
import os
import math
import pickle
import argparse
import pygame
import neat

from track import Track
from car import Car
from simulation import NEATSimulation, SimulationConfig


# ---- Constants ----
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (180, 180, 180)
BLUE = (50, 100, 200)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
YELLOW = (200, 200, 50)
CYAN = (50, 200, 200)
MAGENTA = (200, 50, 200)
ORANGE = (200, 150, 50)
DARK_GREEN = (0, 80, 0)


class App:
    """Main application class for the AI Driving Simulation."""

    def __init__(self, headless=False):
        """Initialize the application."""
        self.headless = headless

        # ---- Initialize pygame ----
        if not headless:
            pygame.init()
            pygame.display.set_caption("AI Learns to Drive")
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.font_small = pygame.font.SysFont('Consolas', 14)
            self.font_medium = pygame.font.SysFont('Consolas', 18)
            self.font_large = pygame.font.SysFont('Arial', 36)
            self.font_huge = pygame.font.SysFont('Arial', 48, bold=True)

        # ---- Game state ----
        self.running = True
        self.paused = False
        self.speed_multiplier = 1.0
        self.fast_forward = False
        self.show_help = False

        # ---- Track ----
        self.track = Track()

        # ---- Cars ----
        self.manual_car = Car(self.track.spawn_x, self.track.spawn_y,
                              self.track.spawn_angle, genome_id='MANUAL')
        self.ai_cars = []            # List of (car, net, genome) tuples
        self.max_ai_cars = 8

        # ---- Input ----
        self.keys = {}
        self.manual_mode = True      # Start in manual mode

        # ---- Camera ----
        self.camera_x = 0
        self.camera_y = 0

        # ---- NEAT ----
        self.config_path = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
        self.simulation = None
        self.training = False

        # ---- Training mode state ----
        self._training_generation = 0
        self._training_best_fitness = float('-inf')
        self._training_avg_fitness = 0.0
        self._training_alive_count = 0
        self._training_total_count = 0
        self._training_finished = False
        self._showcase_best = []     # Best cars from last gen to showcase
        self._showcase_frames_elapsed = 0  # Frames elapsed in current showcase phase

        # ---- Track Editor ----
        self._editor_active = False
        self._editor_drawing = False      # mouse button held down?
        self._editor_waypoints = []      # list of (x, y)
        self._editor_preview = None      # cached preview Track or None

        # ---- UI ----
        self.ui_mode = 'manual'      # 'manual', 'ai_demo', 'training'
        self.show_performance = True

        # ---- Stats ----
        self.frame_count = 0
        self.elapsed_frames = 0

    def run(self):
        """Main application loop."""
        if not self.headless:
            self._main_loop()
        else:
            self._headless_training()

    def _main_loop(self):
        """Main loop with pygame rendering."""
        while self.running:
            effective_fps = FPS * self.speed_multiplier
            dt = self.clock.tick(effective_fps) / 16.667  # Normalize to ~60fps
            self.elapsed_frames += 1

            self._handle_events()
            if not self.paused:
                self._update(dt)
            self._draw()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.keys[event.key] = True

                # ---- Editor keys take priority when editor is active ----
                if self._editor_active:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        self._finalize_editor_track()
                    elif event.key == pygame.K_BACKSPACE:
                        if self._editor_waypoints:
                            self._editor_waypoints.pop()
                            self._editor_preview = None
                    elif event.key == pygame.K_ESCAPE:
                        self._toggle_track_editor()
                    elif event.key == pygame.K_e:
                        self._toggle_track_editor()
                    return  # Don't process game keys while editing

                # ---- Normal game keys ----
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.ui_mode == 'training' and getattr(self, '_gen_phase', None) == 'showcase':
                        self._run_next_generation()
                    else:
                        self.paused = not self.paused
                elif event.key == pygame.K_r:
                    if self.manual_mode:
                        self._reset_manual_car()
                elif event.key == pygame.K_m:
                    self._toggle_mode()
                elif event.key == pygame.K_t:
                    self._toggle_training()
                elif event.key == pygame.K_f:
                    self.fast_forward = not self.fast_forward
                    self.speed_multiplier = 5.0 if self.fast_forward else 1.0
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.speed_multiplier = max(0.25, self.speed_multiplier - 0.5)
                elif event.key == pygame.K_c:
                    self.show_performance = not self.show_performance
                elif event.key == pygame.K_e:
                    self._toggle_track_editor()

            elif event.type == pygame.KEYUP:
                self.keys[event.key] = False

            elif event.type == pygame.MOUSEBUTTONDOWN and self._editor_active:
                mx, my = event.pos
                if event.button == 1:      # Left button → start drawing
                    self._editor_drawing = True
                    self._editor_waypoints.append((mx, my))
                    self._editor_preview = None
                elif event.button == 3:    # Right click → remove last
                    if self._editor_waypoints:
                        self._editor_waypoints.pop()
                        self._editor_preview = None

            elif event.type == pygame.MOUSEMOTION and self._editor_active and self._editor_drawing:
                # Guard: if mouse button was released outside the window, stop drawing
                if not pygame.mouse.get_pressed()[0]:
                    self._editor_drawing = False
                    return
                mx, my = event.pos
                if self._editor_waypoints:
                    lx, ly = self._editor_waypoints[-1]
                    # Only add a point if far enough from the last one
                    dx, dy = mx - lx, my - ly
                    if dx * dx + dy * dy > 100:  # ~10px threshold
                        self._editor_waypoints.append((mx, my))
                        self._editor_preview = None
                else:
                    self._editor_waypoints.append((mx, my))
                    self._editor_preview = None

            elif event.type == pygame.MOUSEBUTTONUP and self._editor_active and event.button == 1:
                self._editor_drawing = False

    def _toggle_track_editor(self):
        """Enter or exit the track editor mode."""
        if self._editor_active:
            # Exiting — discard
            self._editor_active = False
            self._editor_drawing = False
            self._editor_waypoints = []
            self._editor_preview = None
            self.paused = False
            self._status_msg = "Track editor cancelled"
        else:
            # Can't enter while training is running
            if getattr(self, '_gen_phase', None) == 'evaluating':
                return
            self._editor_active = True
            self._editor_drawing = False
            self._editor_waypoints = []
            self._editor_preview = None
            self.paused = True
            self._status_msg = "Click & drag to draw your track · Right-click to undo · Enter to finish"

    def _finalize_editor_track(self):
        """Build a Track from the placed waypoints and switch to it."""
        if len(self._editor_waypoints) < 3:
            self._status_msg = "Need at least 3 waypoints to make a track!"
            return

        try:
            new_track = Track.from_waypoints(self._editor_waypoints)
        except Exception as e:
            self._status_msg = f"Could not build track: {e}"
            return

        self.track = new_track

        # Reset the manual car to the new spawn
        self.manual_car = Car(
            self.track.spawn_x, self.track.spawn_y,
            self.track.spawn_angle, genome_id='MANUAL'
        )

        # Clear AI demo / training state
        self.ai_cars = []
        self._showcase_best = []
        self.training = False
        self.manual_mode = True
        self.ui_mode = 'manual'

        # Exit editor
        self._editor_active = False
        self._editor_drawing = False
        self._editor_waypoints = []
        self._editor_preview = None
        self.paused = False

        self._status_msg = f"Custom track created! ({len(new_track.walls)} walls, {new_track.num_checkpoints} checkpoints)"

    def _toggle_mode(self):
        """Toggle between manual and AI demo mode."""
        if self.manual_mode:
            self.manual_mode = False
            self.ui_mode = 'ai_demo'
            self._setup_ai_demo()
        else:
            self.manual_mode = True
            self.ui_mode = 'manual'
            self.ai_cars = []
            self._reset_manual_car()
        self.training = False

    def _toggle_training(self):
        """Toggle training mode on/off."""
        if self.training:
            self.training = False
            self.manual_mode = True
            self.ui_mode = 'manual'
            self.ai_cars = []
            self._reset_manual_car()
        else:
            self.training = True
            self.manual_mode = False
            self.ui_mode = 'training'
            self._start_training()

    # ──────────────────────────────────────────────
    #  AI Demo Mode
    # ──────────────────────────────────────────────

    def _setup_ai_demo(self):
        """Set up AI demo mode. Loads best genome or shows random networks."""
        self.ai_cars = []
        genome = self._find_best_genome()

        if genome is not None:
            config = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                self.config_path
            )
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            car = Car(self.track.spawn_x, self.track.spawn_y,
                      self.track.spawn_angle, genome_id='BEST')
            self.ai_cars.append((car, net, genome))
            self._status_msg = "Showing trained AI from winner.pkl"
        else:
            # Create a small temporary population to get properly initialized random genomes
            config = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                self.config_path
            )
            temp_pop = neat.Population(config)
            genome_items = list(temp_pop.population.items())
            for i in range(min(4, len(genome_items))):
                genome_id, genome = genome_items[i]
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                car = Car(
                    self.track.spawn_x + i * 15,
                    self.track.spawn_y,
                    self.track.spawn_angle + math.radians(i * 3),
                    genome_id=f'RANDOM-{i}'
                )
                self.ai_cars.append((car, net, genome))
            self._status_msg = "No trained AI found — showing random networks. Run training first!"

    def _find_best_genome(self):
        """Try to find the best saved genome from training."""
        winner_path = 'winner.pkl'
        if os.path.exists(winner_path):
            try:
                with open(winner_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Could not load winner.pkl: {e}")

        checkpoint_dir = 'neat_checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.checkpoint')],
                reverse=True
            )
            if checkpoints:
                try:
                    with open(os.path.join(checkpoint_dir, checkpoints[0]), 'rb') as f:
                        pop = pickle.load(f)
                    return pop.best_genome
                except Exception as e:
                    print(f"Could not load checkpoint: {e}")

        return None

    # ──────────────────────────────────────────────
    #  Training Mode
    # ──────────────────────────────────────────────

    def _start_training(self):
        """Initialize the NEAT simulation for interactive training."""
        self.ai_cars = []

        # Try loading from checkpoint
        checkpoint_dir = 'neat_checkpoints'
        resume_file = None
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.checkpoint')],
                reverse=True
            )
            if checkpoints:
                resume_file = os.path.join(checkpoint_dir, checkpoints[0])

        self.simulation = NEATSimulation(
            self.track, self.config_path,
            visualize=True, headless=False
        )

        if resume_file:
            try:
                self.simulation.load_checkpoint(resume_file)
                print(f"Resumed from checkpoint: {resume_file}")
            except Exception as e:
                print(f"Could not resume: {e}")

        self._showcase_best = []
        self._training_generation = self.simulation.population.generation
        self._gen_phase = 'idle'  # 'idle' | 'evaluating' | 'showcase'

        # Auto-start the first generation so the user sees cars immediately
        self._run_next_generation()

    def _run_next_generation(self):
        """
        Run one full generation through NEAT's proper lifecycle.
        The eval_genomes function will draw frames through a callback.
        """
        if self.simulation is None:
            return

        self._gen_phase = 'evaluating'

        # Store reference to self so the callback can use it
        self.simulation.draw_callback = self._training_draw_callback
        self._training_interrupted = False

        try:
            self.simulation.population.run(self.simulation.eval_genomes, 1)
        except Exception as e:
            print(f"Generation error: {e}")
            self._gen_phase = 'idle'
            return

        # Generation completed — get stats
        gen = self.simulation.population.generation
        self._training_generation = gen

        gen_stats = getattr(self.simulation, '_last_gen_stats', {})
        if gen_stats:
            self._training_best_fitness = max(
                self._training_best_fitness, gen_stats.get('best_fitness', 0)
            )
            self._training_avg_fitness = gen_stats.get('avg_fitness', 0.0)
        else:
            # Fallback: compute from population
            if self.simulation.stats_reporter is not None:
                try:
                    best_fit = max(g.fitness for g in
                                   self.simulation.population.population.values())
                    avg_fit = sum(g.fitness for g in
                                  self.simulation.population.population.values()) / \
                              len(self.simulation.population.population)
                    self._training_best_fitness = max(self._training_best_fitness, best_fit)
                    self._training_avg_fitness = avg_fit
                except Exception:
                    pass

        # Save checkpoint
        os.makedirs('neat_checkpoints', exist_ok=True)
        self.simulation.save_checkpoint(
            f'neat_checkpoints/gen_{gen:04d}.checkpoint'
        )

        # Create showcase of best cars from this generation
        self._setup_showcase_from_population()

        # ---- Print generation summary to terminal (during showcase, not eval) ----
        self._print_gen_summary(gen_stats)

        # Reset showcase timer
        self._showcase_frames_elapsed = 0

        self._gen_phase = 'showcase'

    def _setup_showcase_from_population(self):
        """Create cars from the top-performing genomes to showcase."""
        self._showcase_best = []

        # Get genomes sorted by fitness (filter out any with None fitness)
        genomes = [
            (gid, g) for gid, g in self.simulation.population.population.items()
            if g.fitness is not None
        ]
        genomes.sort(key=lambda x: x[1].fitness, reverse=True)

        # Show top cars
        for genome_id, genome in genomes[:self.max_ai_cars]:
            net = neat.nn.FeedForwardNetwork.create(genome, self.simulation.config)
            car = Car(self.track.spawn_x, self.track.spawn_y,
                      self.track.spawn_angle, genome_id)
            self._showcase_best.append({
                'car': car,
                'net': net,
                'genome': genome,
                'fitness': genome.fitness
            })

        # Reset AI cars display to showcase
        self.ai_cars = [(s['car'], s['net'], s['genome'])
                        for s in self._showcase_best]

    def _training_draw_callback(self, cars, step, generation, alive_count, total_count):
        """
        Called during eval_genomes to render each frame of training.
        This is called from within the blocking population.run() call.
        """
        # Update display cars
        self.ai_cars = [(cars[i], None, None)
                        for i in range(min(len(cars), self.max_ai_cars))]
        self._training_alive_count = alive_count
        self._training_total_count = total_count
        self._training_generation = generation
        self._training_step = step

        # Handle events (allows quitting during training)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self._training_interrupted = True
                return False  # Signal to stop evaluation
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self._training_interrupted = True
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self.fast_forward = not self.fast_forward
                    self.speed_multiplier = 5.0 if self.fast_forward else 1.0
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_c:
                    self.show_performance = not self.show_performance
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.speed_multiplier = max(0.25, self.speed_multiplier - 0.5)
                elif event.key == pygame.K_m:
                    self._pending_exit_training = 'm'
                    return False
                elif event.key == pygame.K_t:
                    self._pending_exit_training = 't'
                    return False

        # Draw the current frame
        self.camera_x = 0
        self.camera_y = 0
        self._draw()
        pygame.display.flip()  # Must flip here since _draw skips it during evaluating

        # Control frame rate
        effective_fps = FPS * self.speed_multiplier
        self.clock.tick(effective_fps)

        # If paused, just wait a bit and keep going (don't abort the generation)
        if self.paused:
            pygame.time.wait(50)
        return self.running  # Only stop if the user quits (Escape/close)

    # ──────────────────────────────────────────────
    #  Update methods
    # ──────────────────────────────────────────────

    def _update(self, dt=1.0):
        """Update game state for one frame."""
        if self.ui_mode == 'manual':
            self._update_manual(dt)
        elif self.ui_mode == 'ai_demo':
            self._update_ai_demo(dt)
        elif self.ui_mode == 'training':
            self._update_training(dt)

        self.frame_count += 1

    def _update_manual(self, dt=1.0):
        """Update manual driving mode."""
        car = self.manual_car
        if not car.alive:
            return

        steering = 0.0
        throttle = 0.0

        if self.keys.get(pygame.K_LEFT, False) or self.keys.get(pygame.K_a, False):
            steering = -1.0
        if self.keys.get(pygame.K_RIGHT, False) or self.keys.get(pygame.K_d, False):
            steering = 1.0
        if self.keys.get(pygame.K_UP, False) or self.keys.get(pygame.K_w, False):
            throttle = 1.0
        if self.keys.get(pygame.K_DOWN, False) or self.keys.get(pygame.K_s, False):
            throttle = -1.0

        car.update(steering, throttle, self.track.walls, dt)
        car.get_checkpoint_progress(self.track.checkpoints)

    def _update_ai_demo(self, dt=1.0):
        """Update AI demo mode — drive cars with their trained networks."""
        for car, net, genome in self.ai_cars:
            if not car.alive:
                continue
            inputs = car.get_inputs()
            outputs = net.activate(inputs)
            steering = max(-1.0, min(1.0, outputs[0]))
            throttle = max(-1.0, min(1.0, outputs[1]))
            car.update(steering, throttle, self.track.walls, dt)
            car.get_checkpoint_progress(self.track.checkpoints)

        # Reset all if all crashed
        all_dead = all(not car.alive for car, _, _ in self.ai_cars)
        if all_dead:
            for car, net, genome in self.ai_cars:
                car.reset(self.track.spawn_x, self.track.spawn_y,
                          self.track.spawn_angle)

    def _update_training(self, dt=1.0):
        """Update training mode."""
        if self.simulation is None:
            return

        # Check for pending mode switch requested during evaluation callback
        pending = getattr(self, '_pending_exit_training', None)
        if pending:
            self._pending_exit_training = None
            self._gen_phase = 'idle'
            if pending == 'm':
                self._toggle_mode()
            elif pending == 't':
                self._toggle_training()
            return

        if self._gen_phase == 'idle' or self._gen_phase == 'showcase':
            # Run showcase cars while waiting
            self._update_showcase(dt)

            # Check if it's time to start next generation
            if self._gen_phase == 'showcase':
                self._showcase_frames_elapsed += 1

                # Check if all showcase cars have crashed OR time limit reached
                all_dead = all(
                    not s['car'].alive for s in self._showcase_best
                )
                time_up = self._showcase_frames_elapsed >= SimulationConfig.SHOWCASE_FRAME_LIMIT
                if (all_dead or time_up) and not self._training_interrupted:
                    self._run_next_generation()

        elif self._gen_phase == 'evaluating':
            # Evaluating — the callback handles everything
            pass

    def _update_showcase(self, dt=1.0):
        """Update the showcase cars for display."""
        for s in self._showcase_best:
            car = s['car']
            if not car.alive:
                continue
            inputs = car.get_inputs()
            outputs = s['net'].activate(inputs)
            steering = max(-1.0, min(1.0, outputs[0]))
            throttle = max(-1.0, min(1.0, outputs[1]))
            car.update(steering, throttle, self.track.walls, dt)
            car.get_checkpoint_progress(self.track.checkpoints)

    def _reset_manual_car(self):
        """Reset the manual car to spawn."""
        self.manual_car.reset(self.track.spawn_x, self.track.spawn_y,
                              self.track.spawn_angle)

    # ──────────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────────

    def _draw(self):
        """Render the entire game scene."""
        self.screen.fill(DARK_GREEN)

        camera_offset = (self.camera_x, self.camera_y)

        # Draw track (current game track behind the editor)
        if not self._editor_active:
            passed_cps = self.manual_car.checkpoints_passed if self.ui_mode == 'manual' else set()
            self.track.draw(self.screen, camera_offset, passed_cps)
        else:
            # During editor, dim the current track
            self.track.draw(self.screen, camera_offset, set())
            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            dim.set_alpha(100)
            dim.fill((0, 0, 0))
            self.screen.blit(dim, (0, 0))

        # Draw manual car (only when not in editor)
        if self.ui_mode == 'manual' and not self._editor_active:
            self.manual_car.draw(self.screen, camera_offset, show_sensors=True)

        # Draw AI cars
        for car, net, genome in self.ai_cars:
            show_sensors = car.alive and (
                self.ui_mode != 'training' or len(self.ai_cars) <= 4
            )
            car.draw(self.screen, camera_offset, show_sensors=show_sensors)

        # Draw track editor overlay if active
        if self._editor_active:
            self._draw_editor_overlay()

        # Overlays
        self._draw_hud()
        if self.show_help:
            self._draw_help()

        if not hasattr(self, '_gen_phase') or self._gen_phase != 'evaluating':
            pygame.display.flip()

    def _draw_editor_overlay(self):
        """Draw the track editor UI overlay."""
        wpts = self._editor_waypoints

        # ---- Generate preview track if enough points ----
        if len(wpts) >= 3:
            if self._editor_preview is None:
                try:
                    self._editor_preview = Track.from_waypoints(wpts)
                except Exception:
                    self._editor_preview = None

            if self._editor_preview is not None:
                preview = self._editor_preview
                ox, oy = 0, 0

                # Draw the preview road surface as individual quads (robust against overlap)
                road_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                n = len(preview._outer_boundary)
                for i in range(n):
                    j = (i + 1) % n
                    quad = [
                        (preview._outer_boundary[i][0] + ox, preview._outer_boundary[i][1] + oy),
                        (preview._outer_boundary[j][0] + ox, preview._outer_boundary[j][1] + oy),
                        (preview._inner_boundary[j][0] + ox, preview._inner_boundary[j][1] + oy),
                        (preview._inner_boundary[i][0] + ox, preview._inner_boundary[i][1] + oy),
                    ]
                    pygame.draw.polygon(road_surf, (60, 60, 60, 160), quad)
                self.screen.blit(road_surf, (0, 0))

                # Draw preview walls
                for wall in preview.walls:
                    p1, p2 = wall
                    pygame.draw.line(self.screen, (200, 50, 50, 200),
                                     (p1[0] + ox, p1[1] + oy),
                                     (p2[0] + ox, p2[1] + oy), 3)

                # Draw preview checkpoints
                for i, cp in enumerate(preview.checkpoints):
                    p1, p2 = cp
                    pygame.draw.line(self.screen, (255, 255, 0, 160),
                                     (p1[0] + ox, p1[1] + oy),
                                     (p2[0] + ox, p2[1] + oy), 3)

                # Draw spawn
                pygame.draw.circle(self.screen, (0, 255, 100),
                                   (int(preview.spawn_x), int(preview.spawn_y)), 6, 2)

        # ---- Draw waypoints ----
        for i, (x, y) in enumerate(wpts):
            # Dot
            pygame.draw.circle(self.screen, (50, 255, 50), (int(x), int(y)), 6)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 8, 1)
            # Number label
            label = self.font_small.render(str(i + 1), True, WHITE)
            self.screen.blit(label, (int(x) + 10, int(y) - 8))

        # ---- Draw lines between waypoints ----
        if len(wpts) >= 2:
            for i in range(len(wpts) - 1):
                p1 = wpts[i]
                p2 = wpts[i + 1]
                pygame.draw.line(self.screen, (50, 255, 50, 120),
                                 p1, p2, 2)
            # Dashed line back to start to show closure
            p1 = wpts[-1]
            p2 = wpts[0]
            Track._draw_dashed_line(self.screen, (50, 255, 50, 80),
                                    p1, p2, 1, 8, 6)

        # ---- Instructions overlay ----
        instr_bg = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
        instr_bg.fill((0, 0, 0, 180))
        self.screen.blit(instr_bg, (0, SCREEN_HEIGHT - 90))

        instr = [
            f"Points: {len(wpts)}  |  Click & drag to draw  |  Right-click: Undo  |  ",
            f"Enter: Finish  |  Esc: Cancel",
        ]
        if len(wpts) >= 3:
            instr[0] += "✓ Valid track"
        else:
            instr[0] += f"Need {3 - len(wpts)} more point(s)"

        for i, line in enumerate(instr):
            text = self.font_small.render(line, True, (200, 200, 100))
            self.screen.blit(text, (20, SCREEN_HEIGHT - 80 + i * 20))

    def _draw_hud(self):
        """Draw heads-up display overlay."""
        # Title bar
        title = self.font_large.render("AI LEARNS TO DRIVE", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 8))

        # Mode badge
        mode_y = 52
        mode_info = {
            'manual':   ('MANUAL DRIVE', BLUE),
            'ai_demo':  ('AI DEMO', CYAN),
            'training': ('AI TRAINING', MAGENTA),
            'track_editor': ('TRACK EDITOR', ORANGE),
        }
        ui_key = 'track_editor' if self._editor_active else self.ui_mode
        mode_label, mode_color = mode_info.get(ui_key, ('UNKNOWN', WHITE))
        badge = self.font_medium.render(mode_label, True, BLACK)
        badge_bg = pygame.Surface((badge.get_width() + 16, badge.get_height() + 6))
        badge_bg.fill(mode_color)
        self.screen.blit(badge_bg, (10, mode_y - 3))
        self.screen.blit(badge, (18, mode_y))

        # Right panel
        if self.show_performance:
            px = SCREEN_WIDTH - 310
            py = 50

            # Background
            perf_bg = pygame.Surface((300, 140), pygame.SRCALPHA)
            perf_bg.fill((0, 0, 0, 160))
            self.screen.blit(perf_bg, (px, py))

            lines = self._get_info_lines()
            for i, line in enumerate(lines):
                text = self.font_small.render(line, True, LIGHT_GRAY)
                self.screen.blit(text, (px + 10, py + 8 + i * 20))

        # Bottom bar
        bar_y = SCREEN_HEIGHT - 44
        bar_bg = pygame.Surface((SCREEN_WIDTH, 44))
        bar_bg.fill((10, 10, 10, 200))
        bar_bg.set_alpha(200)
        self.screen.blit(bar_bg, (0, bar_y))

        bar_texts = []
        if self.paused:
            bar_texts.append("[SPACE] Resume")
        bar_texts += [
            "[H] Help", "[E] Edit Track", "[M] Mode", "[T] Train",
            "[F] Fast", "[/] Slower", "[\\] Faster", "[Esc] Quit"
        ]
        if self.fast_forward:
            bar_texts.insert(0, f">> {self.speed_multiplier:.0f}x <<")

        bar_parts = '  '.join(bar_texts)
        bar_render = self.font_small.render(bar_parts, True, LIGHT_GRAY)
        self.screen.blit(bar_render, (20, bar_y + 14))

        # Status message
        if hasattr(self, '_status_msg') and self._status_msg:
            msg = self.font_small.render(self._status_msg, True, YELLOW)
            self.screen.blit(msg, (20, bar_y - 22))

    def _get_info_lines(self):
        """Get list of info strings for the performance panel."""
        lines = []

        if self.ui_mode == 'manual':
            c = self.manual_car
            lines = [
                f"Speed:   {abs(c.speed)/c.MAX_SPEED*100:5.0f}%",
                f"Steer:   {c.steering:+.2f}",
                f"Dist:    {c.total_distance:6.0f} px",
                f"CP:      {len(c.checkpoints_passed)}/{c.total_checkpoints}",
                f"Status:  {'CRASHED' if c.crashed else 'OK'}",
                f"FPS:     {self.clock.get_fps():.0f}",
            ]

        elif self.ui_mode == 'ai_demo':
            alive = sum(1 for c, _, _ in self.ai_cars if c.alive)
            best_cp = max(
                (len(c.checkpoints_passed) for c, _, _ in self.ai_cars),
                default=0
            )
            lines = [
                f"AI Cars: {len(self.ai_cars)}",
                f"Alive:   {alive}",
                f"Best CP: {best_cp}/{self.track.total_checkpoints}",
                f"Speed:   {self.speed_multiplier:.1f}x",
                f"FPS:     {self.clock.get_fps():.0f}",
            ]

        elif self.ui_mode == 'training':
            gen = self._training_generation
            phase = getattr(self, '_gen_phase', 'idle')
            lines = [
                f"Gen:     {gen}",
                f"Phase:   {phase.upper()}",
                f"Best:    {self._training_best_fitness:8.1f}",
                f"Avg:     {self._training_avg_fitness:8.1f}",
            ]
            if phase == 'evaluating':
                lines += [
                    f"Alive:   {self._training_alive_count}/{self._training_total_count}",
                ]
            elif phase == 'showcase':
                alive = sum(1 for s in self._showcase_best if s['car'].alive)
                remaining = max(0, SimulationConfig.SHOWCASE_FRAME_LIMIT - self._showcase_frames_elapsed)
                remaining_secs = remaining / 60  # Approximate at normal speed
                lines += [
                    f"Shown:   {alive}/{len(self._showcase_best)}",
                    f"Next:    ~{remaining_secs:.0f}s (Space to skip)",
                ]
            lines.append(f"Speed:   {self.speed_multiplier:.1f}x")

        return lines

    def _draw_help(self):
        """Draw the full help overlay."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(210)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        sections = [
            ("CONTROLS", [
                ("Arrow Keys / WASD", "Drive (Manual mode)"),
                ("R", "Reset car"),
                ("E", "Open Track Editor — click to build custom tracks"),
                ("M", "Toggle Manual / AI Demo"),
                ("T", "Toggle Training mode"),
                ("Space", "Pause / Resume"),
                ("[ / ]", "Decrease / Increase speed"),
                ("F", "Toggle fast-forward"),
                ("C", "Toggle performance panel"),
                ("H", "Close help"),
                ("Esc", "Quit"),
            ]),
            ("MODES", [
                ("Manual", "Drive the car yourself with keyboard"),
                ("AI Demo", "Watch a trained AI drive the track"),
                ("Training", "Watch NEAT evolve over generations"),
                ("Track Editor", "Press E, click to place waypoints, Enter to finish"),
            ]),
            ("SENSORS", [
                ("Green rays", "Clear path ahead"),
                ("Yellow rays", "Wall detected at medium range"),
                ("Red rays", "Wall very close — danger!"),
                ("Orange dots", "Ray impact point on wall"),
            ]),
            ("REWARDS", [
                ("Checkpoint", f"+{SimulationConfig.CHECKPOINT_REWARD}"),
                ("Lap bonus", f"+{SimulationConfig.LAP_COMPLETION_BONUS}"),
                ("Collision", f"{SimulationConfig.COLLISION_PENALTY} (fatal)"),
                ("Goal", "Evolve a network that completes laps!"),
            ]),
        ]

        x = 120
        y = 60
        for title, items in sections:
            header = self.font_medium.render(title, True, CYAN)
            self.screen.blit(header, (x, y))
            y += 30

            for label, desc in items:
                label_r = self.font_small.render(f"  {label}", True, GREEN)
                desc_r = self.font_small.render(f"  {desc}", True, LIGHT_GRAY)
                self.screen.blit(label_r, (x + 20, y))
                self.screen.blit(desc_r, (x + 300, y))
                y += 22

            y += 10

        # Close hint
        hint = self.font_medium.render("Press [H] to close help", True, YELLOW)
        self.screen.blit(hint, (SCREEN_WIDTH // 2 - hint.get_width() // 2, SCREEN_HEIGHT - 60))

    # ──────────────────────────────────────────────
    #  Headless Training
    # ──────────────────────────────────────────────

    def _print_gen_summary(self, stats):
        """Print a formatted generation summary to terminal during showcase."""
        if not stats:
            return
        gen = stats.get('generation', '?')
        best = stats.get('best_fitness', 0)
        avg = stats.get('avg_fitness', 0)
        n_species = stats.get('num_species', 0)
        gen_time = stats.get('gen_time', 0)
        pop_size = stats.get('population_size', 0)
        species_rows = stats.get('species_rows', [])

        sep = '═' * 50
        print()
        print(f'  {sep}')
        print(f'    Generation {gen} complete')
        print(f'  {sep}')
        print(f'    Best fitness:     {best:>12.2f}')
        print(f'    Average fitness:  {avg:>12.2f}')
        print(f'    Species:          {n_species:>12d}')
        print(f'    Population:       {pop_size:>12d}')
        print(f'    Generation time:  {gen_time:>8.1f} sec')

        if species_rows:
            print(f'  {"─" * 50}')
            print(f'    {"ID":>4}  {"Age":>3}  {"Size":>4}  {"Fitness":>10}  {"AdjFit":>7}  {"Stag":>4}')
            print(f'    {"──":>4}  {"───":>3}  {"────":>4}  {"──────────":>10}  {"───────":>7}  {"────":>4}')
            for sid, age, members, sp_fit, adj_fit, stag in species_rows:
                print(f'    {sid:>4}  {age:>3}  {members:>4}  {sp_fit:>10.2f}  {adj_fit:>7.3f}  {stag:>4}')
        print(f'  {sep}')
        print()

    def _headless_training(self):
        """Run training without rendering (fastest mode)."""
        print("=" * 60)
        print("  AI LEARNS TO DRIVE — Headless Training")
        print("=" * 60)
        print()

        simulation = NEATSimulation(
            self.track, self.config_path,
            visualize=False, headless=True
        )

        # Add terminal output back for headless mode
        simulation.population.add_reporter(neat.StdOutReporter(True))

        print("Starting evolution...\n")

        # Try resuming from checkpoint
        checkpoint_dir = 'neat_checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.checkpoint')],
                reverse=True
            )
            if checkpoints:
                latest = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Resuming from checkpoint: {latest}")
                try:
                    simulation.load_checkpoint(latest)
                except Exception as e:
                    print(f"  Could not resume: {e}")
                # Re-add StdOutReporter after load_checkpoint clears reporters
                simulation.population.add_reporter(neat.StdOutReporter(True))

        best_genome = simulation.run(num_generations=200,
                                     checkpoint_interval=20)

        if best_genome:
            with open('winner.pkl', 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"\n✓ Best genome saved to 'winner.pkl'")
            print(f"  Fitness: {best_genome.fitness:.2f}")

        print("\nTraining complete!")

        stats = simulation.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Generations: {stats['generation']}")
        if stats['fitness_history']:
            best_fits = [f[0] for f in stats['fitness_history']]
            print(f"  Peak fitness: {max(best_fits):.2f}")

    def cleanup(self):
        """Clean up resources."""
        if not self.headless:
            pygame.quit()


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

def main():
    """Parse args and run the application."""
    parser = argparse.ArgumentParser(
        description='AI Learns to Drive — NEAT-powered driving AI'
    )
    parser.add_argument('--headless', action='store_true',
                        help='Run training without rendering (fastest)')
    parser.add_argument('--train', action='store_true',
                        help='Start in interactive training mode')
    parser.add_argument('--demo', action='store_true',
                        help='Start in AI demo mode')

    args = parser.parse_args()

    app = App(headless=args.headless)

    if args.train:
        app.training = True
        app.manual_mode = False
        app.ui_mode = 'training'
        app._start_training()
    elif args.demo:
        app.manual_mode = False
        app.ui_mode = 'ai_demo'
        app._setup_ai_demo()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        app.cleanup()
        print("Goodbye!")


if __name__ == '__main__':
    main()
