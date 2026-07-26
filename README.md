# AI Learns to Drive

A reinforcement learning project where an AI learns to drive a car around a race track using **NEAT** (NeuroEvolution of Augmenting Topologies). The car navigates using ray-cast sensors and evolves its neural network over generations to master the circuit.

![AI Learns to Drive](screenshot.png)

> **No training data. No human demonstrations. Just raw evolution.**  
> The AI starts knowing absolutely nothing about driving — and over generations learns to steer, accelerate, and follow the track entirely on its own.

---

## Features

- **NEAT-Python Evolution** — NeuroEvolution of Augmenting Topologies automatically designs and optimizes neural networks
- **Custom Race Track** — A flowing, closed-loop circuit with wide corners, an S-bend, and a large hairpin
- **Ray-Cast Sensors** — 7 directional sensors let the car "see" walls up to 400px away
- **Manual Drive Mode** — Take the wheel yourself with keyboard controls
- **AI Demo Mode** — Watch a trained or random AI drive the track
- **Interactive Training** — Watch NEAT evolve generation by generation in real-time
- **Headless Training** — Fast, no-rendering mode for serious training sessions
- **Track Editor** — Draw custom tracks by clicking waypoints (press `E`)
- **Real-time Visualization** — Sensor rays, checkpoint tracking, species statistics
- **Checkpoint Rewards** — The AI earns fitness for passing checkpoints, encouraging forward progress

---

## How It Works

### The Car

Each car has:
- **7 ray-cast sensors** spread across ±60° in front of the car
- A **feedforward neural network** with 8 inputs (7 sensor distances + speed) and 2 outputs (steering, throttle)
- Simple physics: acceleration, friction, speed-dependent turning radius
- Collision detection against track walls (instantly fatal)

### The Brain (NEAT)

[NEAT](https://neat-python.readthedocs.io/) evolves both the weights *and the structure* of the neural network:
- Starts with minimal networks (no hidden nodes)
- Over generations, adds nodes and connections through mutation
- Keeps useful innovations through speciation
- Rewards cars that navigate further, pass checkpoints, and stay on track

### The Track

A single continuous closed-loop circuit designed for smooth, flowing driving:

```
         ┌────────────────────────────────────┐
        ╱                                      ╲
       │   Top Straight (long)                  │
       │                                        │
       │    ┌──────────┐                        │
       │    │  S-bend   ╲                       │
       │    │  (centre)  │          Hairpin     │
       │    └──────────┘  ╲         (far-right) │
       ╲                   ╲                    ╱
        ╲    Start ◇        ╲                  ╱
         └────────────────────┘────────────────┘
```

| Feature | Direction | Description |
|---|---|---|
| Long diagonal straight | Upper-left | From start at bottom-centre |
| Sweeping left turn | Top-left | Gentle, broad corner |
| Long straight | Rightward | Across the top of the circuit |
| Right curve | Descending | Wide right-hand bend toward centre |
| S-bend | Centre | Sharp leftward zig-zag chicane |
| Short straight | Rightward | Brief straight after the chicane |
| Gradual right climb | Upper-right | Climbing bend toward the far side |
| Right-hand hairpin | Far-right | Long sweeping 180°+ turn |
| Descending curve | Lower-right | Flowing descent |
| Closing left curve | Bottom | Broad left-hand sweep back to start |

### The Sensors

The car shoots 7 rays in a fan pattern in front of it (at angles: -60°, -35°, -15°, 0°, +15°, +35°, +60°). Each ray measures the distance to the nearest wall, normalized to a 0–1 value. The network uses these distances to decide how to steer and accelerate.

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-learns-to-drive

# Install dependencies
pip install -r requirements.txt
```

If a `requirements.txt` doesn't exist yet, install manually:

```bash
pip install pygame neat-python
```

### Run

```bash
# Start in manual drive mode (default)
python main.py

# Start in AI demo mode
python main.py --demo

# Start in interactive training mode
python main.py --train

# Headless training (fastest — no rendering)
python main.py --headless
```

---

## Controls

| Key | Action |
|---|---|
| `Arrow Keys` / `W A S D` | Drive (Manual mode) |
| `R` | Reset car to spawn (Manual mode) |
| `M` | Toggle Manual / AI Demo mode |
| `T` | Toggle Training mode |
| `Space` | Pause / Resume (or start next generation in Training) |
| `[` / `]` | Decrease / Increase simulation speed |
| `F` | Toggle fast-forward (5× speed) |
| `E` | Open Track Editor |
| `C` | Toggle performance panel |
| `H` | Toggle help overlay |
| `Esc` | Quit |

### Track Editor (`E`)

1. Click and drag to place waypoints on the canvas
2. Right-click to undo the last waypoint
3. Press `Enter` to build the track from your waypoints
4. Press `Esc` to cancel

---

## Project Structure

```
ai-learns-to-drive/
├── main.py              # Entry point — App class, UI, controls, modes
├── track.py             # Track class — geometry, walls, checkpoints, drawing
├── car.py               # Car class — physics, sensors, collision, drawing
├── simulation.py        # NEATSimulation — training loop, fitness calculation
├── neat_config.txt      # NEAT hyperparameter configuration
├── winner.pkl           # Saved best genome (generated by training)
├── neat_checkpoints/    # Training checkpoints (generated)
└── README.md            # This file
```

### Module Details

| Module | Purpose |
|---|---|
| `main.py` | Application lifecycle, pygame window, event handling, UI modes, camera |
| `track.py` | Track geometry (centre line, walls, boundaries), checkpoint system, rendering |
| `car.py` | Car physics, sensor ray-casting, collision detection, drawing with rotation |
| `simulation.py` | NEAT integration, per-generation evaluation, fitness calculation, stats |

---

## NEAT Configuration

The `neat_config.txt` file controls evolution parameters. Key settings:

| Parameter | Value | Description |
|---|---|---|
| `pop_size` | 150 | Number of cars per generation |
| `num_hidden` | 0 (min) / 6 (max) | Network complexity bounds |
| `num_inputs` | 8 | 7 sensors + speed |
| `num_outputs` | 2 | Steering & throttle |
| `weight_mutate_power` | 0.6 | Mutation severity |
| `compatibility_threshold` | 3.0 | Species clustering |

---

## Fitness Function

The AI earns fitness through:

| Event | Reward/Penalty |
|---|---|
| Moving forward | +0.5 per unit speed |
| Maintaining speed | +0.1 per unit speed |
| Passing a checkpoint | +200.0 |
| Completing a lap | +500.0 + 1000.0 completion bonus |
| Colliding with a wall | -500.0 (instant death) |
| Driving backwards | -5.0 per frame |
| Staying still | -1.0 per frame |
| Driving in circles | -0.5 per frame |
| Time penalty | -0.01 per frame |

---

## Training Tips

1. **Start with headless mode** (`--headless`) for rapid evolution — it's 10–50× faster
2. **Resume automatically** — checkpoints are saved every generation in `neat_checkpoints/`
3. **Interactive training** (`--train` or press `T`) lets you watch the AI improve visually
4. **Use fast-forward** (`F`) during interactive training to speed through early generations
5. **The best genome** is saved to `winner.pkl` after headless training completes
6. **Track editor** (`E`) lets you design custom circuits to test generalization

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- [NEAT-Python](https://neat-python.readthedocs.io/) — NeuroEvolution of Augmenting Topologies library
- [Pygame](https://www.pygame.org/) — Python game development library
- Inspired by the classic "AI learns to drive" experiments in evolutionary computing
