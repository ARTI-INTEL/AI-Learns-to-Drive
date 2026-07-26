"""
Simulation module for the AI Driving Simulation.

Handles the NEAT training loop: running cars, calculating fitness,
and evolving the population over generations.
"""

import math
import os
import pickle
import time
import neat
from car import Car
from track import Track


class SimulationConfig:
    """Configuration parameters for the simulation."""

    # Time limits
    MAX_STEPS_PER_GEN = 1000        # Max frames per generation
    STUCK_TIME_LIMIT = 150          # Frames before car is considered stuck
    LAPS_FOR_COMPLETION = 1         # Laps needed to "finish"

    # Rewards and punishments (user's recommended values)
    CHECKPOINT_REWARD = 200.0       # Reward for passing a checkpoint
    LAP_COMPLETION_BONUS = 500.0    # Bonus for completing a lap
    DISTANCE_REWARD_FACTOR = 0.5    # Reward per unit distance moved forward
    SPEED_REWARD_FACTOR = 0.1       # Reward for maintaining speed
    COLLISION_PENALTY = -500.0      # Penalty for crashing (user: -500)
    STUCK_PENALTY = -1.0            # Per-frame penalty when staying still (user: -1)
    BACKWARDS_PENALTY = -5.0        # Per-frame penalty for driving backwards (user: -5)
    CIRCLES_PENALTY = -0.5          # Per-frame penalty for driving in circles (user: -0.5)
    TIME_PENALTY = -0.01            # Small per-frame penalty to encourage efficiency
    OFF_TRACK_BONUS = -0.1          # Bonus per sensor pointing at wall (encourages staying on track)

    # Visualization
    GENERATIONS_TO_SHOW = 50        # Max generations to show in window at normal speed
    FAST_FORWARD_AFTER = 100        # Auto fast-forward after this many generations


class NEATSimulation:
    """
    Manages the NEAT evolutionary training process.

    Creates cars controlled by NEAT genomes, evaluates their fitness
    by running them around the track, and returns fitness scores to NEAT.
    """

    def __init__(self, track, config_path, visualize=False, headless=False):
        """
        Initialize the simulation.

        Args:
            track: Track object with walls and checkpoints
            config_path: Path to NEAT config file
            visualize: Whether to show pygame visualization during training
            headless: Whether to run without any rendering (fastest)
        """
        self.track = track
        self.visualize = visualize
        self.headless = headless

        # Load NEAT configuration
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        # Create the population
        self.population = neat.Population(self.config)

        # Add reporter for historical stats (no terminal output during eval)
        self.stats_reporter = neat.StatisticsReporter()
        self.population.add_reporter(self.stats_reporter)

        # Current state
        self.generation = 0
        self.best_genome = None
        self.best_fitness = float('-inf')

        # For visualization
        self.cars = []
        self.max_cars_shown = 50  # Don't try to render all 150 cars

        # Track history
        self.fitness_history = []
        self.species_history = []

        # Last generation stats (printed during showcase instead of during eval)
        self._last_gen_stats = {}
        self._gen_start_time = None

    def eval_genomes(self, genomes, config):
        """
        Evaluate all genomes in the current generation.
        This is the function NEAT calls to calculate fitness.

        Args:
            genomes: List of (genome_id, genome) tuples from NEAT
            config: NEAT configuration
        """
        self.generation += 1
        self._gen_start_time = time.time()

        # Create neural networks and cars for each genome
        nets = []
        self.cars = []

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            car = Car(
                self.track.spawn_x,
                self.track.spawn_y,
                self.track.spawn_angle,
                genome_id
            )
            nets.append(net)
            self.cars.append(car)
            genome.fitness = 0.0

        # Run the simulation
        self._run_simulation(nets, car_indices=range(len(genomes)))

        # Assign fitness back to genomes
        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = self.cars[i].fitness

        # Collect stats for terminal output during showcase
        fitnesses = [g.fitness for _, g in genomes if g.fitness is not None]
        if fitnesses:
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
        else:
            best_fit = 0.0
            avg_fit = 0.0

        # Species info
        num_species = len(self.population.species.species)
        species_rows = []
        for sid, sp in self.population.species.species.items():
            members = len(sp.members)
            age = self.population.generation - sp.created
            sp_fitness = sp.fitness if sp.fitness is not None else 0.0
            adj_fit = sp.adjusted_fitness if sp.adjusted_fitness is not None else 0.0
            stag = self.population.generation - sp.last_improved
            species_rows.append((sid, age, members, sp_fitness, adj_fit, stag))

        self._last_gen_stats = {
            'generation': self.generation,
            'best_fitness': best_fit,
            'avg_fitness': avg_fit,
            'num_species': num_species,
            'gen_time': time.time() - self._gen_start_time,
            'population_size': len(genomes),
            'species_rows': species_rows,
        }

        # Track history
        if fitnesses:
            self.fitness_history.append((best_fit, avg_fit))
            self.species_history.append(num_species)

    def _run_simulation(self, nets, car_indices):
        """Run the simulation for all cars for up to MAX_STEPS_PER_GEN frames."""
        step = 0
        running = len(car_indices)

        while running > 0 and step < SimulationConfig.MAX_STEPS_PER_GEN:
            step += 1

            for i in car_indices:
                car = self.cars[i]
                if not car.alive:
                    continue

                # Get inputs from car's sensors
                inputs = car.get_inputs()

                # Get outputs from neural network
                outputs = nets[i].activate(inputs)

                # Map outputs to car controls
                steering = max(-1.0, min(1.0, outputs[0]))
                throttle = max(-1.0, min(1.0, outputs[1]))

                # Update car physics and cast sensors
                car.update(steering, throttle, self.track.walls)

                # ---- Calculate fitness ----
                self._update_fitness(car, step)

                # ---- Check if car is done ----
                if not car.alive:
                    running -= 1
                elif car.time_stuck > SimulationConfig.STUCK_TIME_LIMIT:
                    car.alive = False
                    car.fitness += SimulationConfig.STUCK_PENALTY * SimulationConfig.STUCK_TIME_LIMIT
                    running -= 1

            # ---- Call the draw callback if set (for interactive training) ----
            if hasattr(self, 'draw_callback') and self.draw_callback is not None:
                should_continue = self.draw_callback(
                    self.cars, step, self.generation, running, len(car_indices)
                )
                if should_continue is False:
                    # Allow early termination (e.g., user pressed Escape)
                    # Kill all remaining cars
                    for i in car_indices:
                        car = self.cars[i]
                        if car.alive:
                            car.alive = False
                    break

    def _update_fitness(self, car, step):
        """Calculate and update the car's fitness based on its performance."""
        # ---- Living reward: reward movement (Step 9) ----
        if car.speed > 0.1:
            # Reward moving forward in the direction the car is facing
            car.fitness += car.speed * SimulationConfig.DISTANCE_REWARD_FACTOR

        # ---- Speed reward ----
        car.fitness += abs(car.speed) * SimulationConfig.SPEED_REWARD_FACTOR

        # ---- Punishments ----
        # Driving backwards (Step 8)
        if car.speed < -0.5:
            car.fitness += SimulationConfig.BACKWARDS_PENALTY

        # Staying still (Step 8)
        if car.time_stuck > 30:
            car.fitness += SimulationConfig.STUCK_PENALTY

        # Driving in circles - detect by summing recent angles (Step 8)
        # Simple heuristic: if car hasn't significantly changed position but has turned a lot
        if hasattr(car, 'total_angle_change'):
            if car.total_angle_change > 6.28 and car.total_distance < 50:
                car.fitness += SimulationConfig.CIRCLES_PENALTY

        # Small time penalty (encourages efficiency)
        car.fitness += SimulationConfig.TIME_PENALTY

        # ---- Checkpoint rewards (Step 7) ----
        cp_passed = car.get_checkpoint_progress(self.track.checkpoints)
        if cp_passed >= 0:
            car.fitness += SimulationConfig.CHECKPOINT_REWARD
            # Bonus for passing checkpoints quickly
            speed_bonus = max(0, 50 - step * 0.1) if car.speed > 0 else 0
            car.fitness += speed_bonus

            # If all 4 checkpoints passed, complete a lap
            if len(car.checkpoints_passed) >= self.track.total_checkpoints:
                car.lap += 1
                car.fitness += SimulationConfig.LAP_COMPLETION_BONUS
                car.checkpoints_passed.clear()  # Reset for next lap

                if car.lap >= SimulationConfig.LAPS_FOR_COMPLETION:
                    car.finished = True
                    car.alive = False
                    # Bonus for finishing
                    car.fitness += 1000.0

        # ---- Collision penalty (Step 8) ----
        if car.crashed and not car.finished:
            car.fitness += SimulationConfig.COLLISION_PENALTY

    def run(self, num_generations=100, checkpoint_interval=10):
        """
        Run the NEAT evolution for the specified number of generations.

        Args:
            num_generations: Number of generations to run
            checkpoint_interval: Save checkpoint every N generations

        Returns:
            The best genome found
        """
        # Create checkpoint directory
        os.makedirs('neat_checkpoints', exist_ok=True)

        try:
            best_genome = self.population.run(self.eval_genomes, num_generations)
            self.best_genome = best_genome
            return best_genome
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            # Return best so far
            return self.population.best_genome

    def get_best_network(self, genome=None):
        """
        Create a feed-forward network from the best genome.

        Args:
            genome: Optional specific genome to use. If None, uses the population's best.

        Returns:
            The neural network and genome
        """
        if genome is None:
            genome = self.population.best_genome
        if genome is None:
            return None, None

        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return net, genome

    def get_stats(self):
        """
        Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        stats = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'species_history': self.species_history,
        }

        if self.stats_reporter and hasattr(self.stats_reporter, 'best_fitness'):
            try:
                stats['best_fitness'] = self.stats_reporter.best_fitness()
            except Exception:
                pass

        return stats

    def save_checkpoint(self, filename=None):
        """Save a NEAT checkpoint to resume later."""
        os.makedirs('neat_checkpoints', exist_ok=True)
        if filename is None:
            filename = f'neat_checkpoints/gen_{self.generation:04d}.checkpoint'
        # Pickle the population directly (reliable across NEAT-Python versions)
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)
        return filename

    def load_checkpoint(self, filename):
        """Load a NEAT checkpoint."""
        with open(filename, 'rb') as f:
            self.population = pickle.load(f)
        self.generation = self.population.generation
        # Clear any reporters that came with the pickled population,
        # then add fresh ones (no StdOutReporter — stats printed during showcase)
        self.population.reporters = neat.reporting.ReporterSet()
        self.stats_reporter = neat.StatisticsReporter()
        self.population.add_reporter(self.stats_reporter)
        self._last_gen_stats = {}
        self._gen_start_time = None
        return self.population
