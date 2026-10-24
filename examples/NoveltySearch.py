#!/usr/bin/env python3
"""Novelty search vs. fitness search in a deceptive maze.

A small robot (controlled by an evolved network) starts in the bottom right corner
of a maze and has to reach the ball in the top right corner. The walls make the
straightforward fitness - "get close to the ball" - deceptive: moving towards the
ball leads into a dead end.

Novelty search ignores the objective and rewards robots that end up somewhere new
(the behavior is the robot's final position). It usually finds the ball much
faster than the fitness based search.

The world is simulated with MultiNEAT's built-in physics engine (MultiNEAT.physics2d).

usage:
    python NoveltySearch.py            # novelty search, no graphics
    python NoveltySearch.py --fitness  # objective (fitness) search for comparison
    python NoveltySearch.py --show     # watch the robots (needs pygame), press F for fast mode
"""
import argparse
import math
import pickle
import random as rnd

import MultiNEAT as NEAT
from MultiNEAT import physics2d as phys

# novelty search settings
ns_K = 15
ns_recompute_sparseness_each = 20
ns_P_min = 10.0
ns_dynamic_Pmin = True
ns_Pmin_min = 1.0
ns_no_archiving_stagnation_threshold = 150
ns_Pmin_lowering_multiplier = 0.9
ns_Pmin_raising_multiplier = 1.1
ns_quick_archiving_min_evals = 8

screen_size_x, screen_size_y = 600, 600
max_timesteps = 1200
solved_distance = 30

collision_type_wall = 0
collision_type_nn = 1
collision_type_ball = 2

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.AllowClones = False
params.AllowLoops = True
params.CompatTreshold = 5.0
params.CompatTresholdModifier = 0.3
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 3
params.MaxSpecies = 10
params.RouletteWheelSelection = True
params.RecurrentProb = 0.2
params.OverallMutationRate = 0.02
params.MutateWeightsProb = 0.90
params.WeightMutationMaxPower = 1.0
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.01
params.WeightMutationRate = 0.75
params.MaxWeight = 8
params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.02
params.MutateRemLinkProb = 0.00
params.EliteFraction = 0.1
params.CrossoverRate = 0.5
params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0


def make_maze():
    space = phys.Space()
    space.gravity = (0.0, 0.0)

    static = phys.Body.static()
    walls = [  # the enclosure
        phys.Segment(static, (50, 50), (50, 550), 5),
        phys.Segment(static, (50, 550), (560, 550), 5),
        phys.Segment(static, (560, 550), (560, 50), 5),
        phys.Segment(static, (50, 50), (560, 50), 5),

        # the obstacle walls
        phys.Segment(static, (120, 480), (560, 480), 5),
        phys.Segment(static, (180, 480), (180, 180), 5),
        phys.Segment(static, (320, 50), (320, 360), 5),
        phys.Segment(static, (440, 480), (440, 360), 5),
    ]
    for s in walls:
        s.friction = 0
        s.elasticity = 0.99
        s.collision_type = collision_type_wall
    space.add(static, walls)
    return space


class NN_agent:
    def __init__(self, space, brain):
        self.body = phys.Body(500)
        self.body.position = (480, 80)
        self.body.velocity_limit = 300
        self.shape = phys.Circle(self.body, 4)
        self.shape.collision_type = collision_type_nn
        self.shape.elasticity = 1.0
        space.add(self.body, self.shape)
        self.brain = brain

    def interact(self, ball):
        """
        inputs: x - ball_x, y - ball_y, self_vx, self_vy, 1
        output: x velocity [-1 .. 1]*const, y velocity [-1 .. 1]*const
        """
        p, b, v = self.body.position, ball.body.position, self.body.velocity
        self.brain.Input([(p.x - b.x) / 300, (p.y - b.y) / 300, v.x / 300, v.y / 300, 1.0])
        self.brain.Activate()
        out = self.brain.Output()
        self.body.velocity = (out[0] * 200, out[1] * 200)


class Ball:
    def __init__(self, space):
        self.body = phys.Body(1500)
        self.body.position = (520, 520)
        self.body.velocity_limit = 5
        self.shape = phys.Circle(self.body, 10)
        self.shape.collision_type = collision_type_ball
        self.shape.elasticity = 1.0
        space.add(self.body, self.shape)


class Behavior:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)


class Display:
    """Optional pygame window."""

    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_size_x, screen_size_y))
        pygame.display.set_caption("Novelty Search [Press F to turn on/off fast mode]")
        self.clock = pygame.time.Clock()
        self.fast_mode = True

    def update(self, space, archive=()):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise SystemExit
            if event.type == pg.KEYDOWN and event.key == pg.K_f:
                self.fast_mode = not self.fast_mode
        if self.fast_mode:
            return
        self.screen.fill((0, 0, 0))
        for b in archive:
            pg.draw.circle(self.screen, (60, 60, 0), (int(b.x), int(screen_size_y - b.y)), 2)
        phys.draw_pygame(self.screen, space, colors={collision_type_ball: (255, 80, 80),
                                                     collision_type_nn: (80, 255, 80)})
        pg.display.flip()
        self.clock.tick(50)


def evaluate(genome, space, display=None, archive=()):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    agent = NN_agent(space, net)
    ball = Ball(space)

    best_distance = 1000000
    dt = 1.0 / 50.0
    for tstep in range(max_timesteps):
        space.step(dt)

        # The NN interacts with the world on each 5 timesteps
        if (tstep % 5) == 0:
            agent.interact(ball)

        if display:
            display.update(space, archive)

        d = (ball.body.position - agent.body.position).length()
        best_distance = min(best_distance, d)

    fitness = 10000 - best_distance

    # remove objects from space
    space.remove(agent.shape, agent.body, ball.shape, ball.body)

    return fitness, best_distance, Behavior(agent.body.position.x, agent.body.position.y)


def main():
    global ns_P_min

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fitness', action='store_true', help='objective search instead of novelty search')
    parser.add_argument('--show', action='store_true', help='show the simulation (needs pygame)')
    parser.add_argument('--evaluations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    display = Display() if args.show else None
    space = make_maze()

    g = NEAT.Genome(0, 5, 0, 2, False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    seed = args.seed if args.seed is not None else rnd.randint(0, 1000)
    pop = NEAT.Population(g, params, True, 1.0, seed)

    archive = []
    best_ever = 0
    hof = []

    def sparseness(genome):
        distances = [genome.behavior.distance_to(g.behavior) for g in NEAT.GetGenomeList(pop)]
        distances += [genome.behavior.distance_to(ab) for ab in archive]
        distances.sort()
        nearest = distances[1:ns_K + 1]
        return sum(nearest) / len(nearest)

    print("Please wait for the initial evaluation to complete.")
    for genome in NEAT.GetGenomeList(pop):
        fitness, _, behavior = evaluate(genome, space, display)
        genome.behavior = behavior
        genome.SetFitness(fitness)
        genome.SetEvaluated()
    if not args.fitness:
        for genome in NEAT.GetGenomeList(pop):
            genome.SetFitness(sparseness(genome))

    evals_since_last_archiving = 0
    quick_add_counter = 0

    for evaluation in range(1, args.evaluations + 1):
        if (not args.fitness) and (evaluation % ns_recompute_sparseness_each == 0):
            for genome in NEAT.GetGenomeList(pop):
                genome.SetFitness(sparseness(genome))

        # replace the worst individual with a new baby
        new = pop.Tick(NEAT.Genome())

        fitness, distance, behavior = evaluate(new, space, display, archive)
        new.behavior = behavior

        if args.fitness:
            new.SetFitness(fitness)
        else:
            sp = sparseness(new)

            # add behavior to archive if above threshold
            evals_since_last_archiving += 1
            if sp > ns_P_min:
                archive.append(new.behavior)
                evals_since_last_archiving = 0
                quick_add_counter += 1
            else:
                quick_add_counter = 0

            if ns_dynamic_Pmin:
                if evals_since_last_archiving > ns_no_archiving_stagnation_threshold:
                    ns_P_min = max(ns_P_min * ns_Pmin_lowering_multiplier, ns_Pmin_min)
                    evals_since_last_archiving = 0
                # too much additions one after another?
                if quick_add_counter > ns_quick_archiving_min_evals:
                    ns_P_min *= ns_Pmin_raising_multiplier
                    quick_add_counter = 0

            new.SetFitness(sp)
        new.SetEvaluated()

        # still use the objective to know which genome is best
        if fitness > best_ever:
            best_ever = fitness
            hof.append(pickle.dumps(new))
            print('Evaluations: %d Species: %d Distance to the ball: %.1f Archive: %d' % (
                evaluation, len(pop.Species), distance, len(archive)))

        if distance < solved_distance:
            print('The robot reached the ball after %d evaluations!' % evaluation)
            break
    else:
        print('Not solved in %d evaluations' % args.evaluations)

    if display:
        # Show the best genome's performance forever
        display.fast_mode = False
        display.pygame.display.set_caption("Best genome ever")
        best = pickle.loads(hof[-1])
        while True:
            evaluate(best, space, display)


if __name__ == '__main__':
    main()
