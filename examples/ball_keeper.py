#!/usr/bin/env python3
"""NEAT ball keeper - evolve a robot that keeps a bouncing ball in the air.

The robot (the green circle) can run left/right and jump. The ball falls under
gravity and bounces off the robot and the walls. An evaluation ends when the ball
touches the floor; the fitness is the number of time steps the ball stayed up.

The world is simulated with MultiNEAT's built-in physics engine (MultiNEAT.physics2d).

usage:
    python ball_keeper.py            # evolve without graphics
    python ball_keeper.py --show     # watch (needs pygame). F toggles fast mode,
                                     # the arrow keys push the ball
"""
import argparse
import math
import random as rnd
import time

import MultiNEAT as NEAT
from MultiNEAT import physics2d as phys

collision_type_wall = 0
collision_type_nn = 1
collision_type_ball = 2
collision_type_floor = 3

screen_size_x, screen_size_y = 600, 600

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.AllowClones = False
params.CompatTreshold = 5.0
params.CompatTresholdModifier = 0.3
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 3
params.MaxSpecies = 10
params.RouletteWheelSelection = True
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.02
params.MutateWeightsProb = 0.90
params.WeightMutationMaxPower = 1.0
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.01
params.WeightMutationRate = 0.75
params.MaxWeight = 20
params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.02
params.MutateRemLinkProb = 0.00
params.EliteFraction = 0.1
params.CrossoverRate = 0.5
params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0


def make_world():
    space = phys.Space()
    space.gravity = (0.0, -500.0)

    static = phys.Body.static()
    # walls - the left-top-right walls
    walls = [phys.Segment(static, (50, 50), (50, 1550), 10),
             phys.Segment(static, (50, 1550), (560, 1550), 10),
             phys.Segment(static, (560, 1550), (560, 50), 10)]
    for s in walls:
        s.friction = 0
        s.elasticity = 0.99
        s.collision_type = collision_type_wall

    floor = phys.Segment(static, (50, 50), (560, 50), 10)
    floor.friction = 1.0
    floor.elasticity = 0.0
    floor.collision_type = collision_type_floor

    space.add(static, walls, floor)
    return space


class NN_agent:
    def __init__(self, space, brain, start_x):
        self.body = phys.Body(50000)
        self.body.position = (start_x, 80)
        self.body.velocity = (230, 0)
        self.body.velocity_limit = 1500
        self.shape = phys.Circle(self.body, 20)
        self.shape.collision_type = collision_type_nn
        self.shape.elasticity = 1.0
        space.add(self.body, self.shape)

        self.brain = brain
        self.in_air = False

    def jump(self):
        if not self.in_air:
            self.body.velocity = (self.body.velocity.x, 300)

    def move(self, x):
        if not self.in_air:
            self.body.velocity = (x, self.body.velocity.y)

    def interact(self, ball):
        """
        inputs: x - ball_x, log(y), ball_vx, ball_vy, in_air, 1
        output: x velocity [-1 .. 1]*const, jump (if > 0.5 )
        """
        p, b, bv = self.body.position, ball.body.position, ball.body.velocity
        inputs = [(p.x - b.x) / 300,
                  math.log(max(p.y, 1.0)),
                  bv.x / 300,
                  bv.y / 300,
                  1.0 if self.in_air else 0.0,
                  1.0]

        self.brain.Input(inputs)
        self.brain.Activate()
        outputs = self.brain.Output()

        self.move(outputs[0] * 500)
        if outputs[1] > 0.5:
            self.jump()


class Ball:
    def __init__(self, space, start_x, start_vx):
        self.body = phys.Body(1500)
        self.body.position = (start_x, 450)
        self.body.velocity = (start_vx, 0)
        self.body.velocity_limit = 500
        self.shape = phys.Circle(self.body, 30)
        self.shape.collision_type = collision_type_ball
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        space.add(self.body, self.shape)
        self.in_air = True


class Display:
    """Optional pygame window."""

    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_size_x, screen_size_y))
        pygame.display.set_caption("NEAT ball keeper [F - fast mode on/off, arrow keys move the ball]")
        self.clock = pygame.time.Clock()
        self.fast_mode = True

    def update(self, space, ball):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise SystemExit
            if event.type == pg.KEYDOWN:
                v = ball.body.velocity
                if event.key == pg.K_f:
                    self.fast_mode = not self.fast_mode
                elif event.key == pg.K_LEFT and not self.fast_mode:
                    ball.body.velocity = (v.x - 200, v.y)
                elif event.key == pg.K_RIGHT and not self.fast_mode:
                    ball.body.velocity = (v.x + 200, v.y)
                elif event.key == pg.K_UP and not self.fast_mode:
                    ball.body.velocity = (v.x, v.y + 200)
        if self.fast_mode:
            return
        self.screen.fill((0, 0, 0))
        phys.draw_pygame(self.screen, space, colors={collision_type_nn: (80, 255, 80),
                                                     collision_type_ball: (255, 80, 80)})
        pg.display.flip()
        self.clock.tick(50)


def evaluate(genome, space, max_timesteps, start_x, start_vx, bot_startx, display=None):
    # The agents - the brain and the ball
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    agent = NN_agent(space, net, bot_startx)
    ball = Ball(space, start_x, start_vx)

    def touch_floor(obj):
        def f(a, b):
            obj.in_air = False
        return f

    def leave_floor(obj):
        def f(a, b):
            obj.in_air = True
        return f

    space.add_collision_handler(collision_type_nn, collision_type_floor, touch_floor(agent), leave_floor(agent))
    space.add_collision_handler(collision_type_ball, collision_type_floor, touch_floor(ball), leave_floor(ball))

    dt = 1.0 / 50.0
    tstep = 0
    while tstep < max_timesteps:
        tstep += 1
        space.step(dt)

        # The NN interacts with the world on each 20 timesteps
        if (tstep % 20) == 0:
            agent.interact(ball)

        # stopping conditions
        if not ball.in_air:
            break

        if display:
            display.update(space, ball)

    fitness = tstep
    if ball.body.position.y < 0:
        fitness = 0

    # remove objects from space
    space.remove(agent.shape, agent.body, ball.shape, ball.body)
    space.remove_collision_handler(collision_type_nn, collision_type_floor)
    space.remove_collision_handler(collision_type_ball, collision_type_floor)

    return fitness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--show', action='store_true', help='show the simulation (needs pygame)')
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=5, help='evaluations per genome')
    parser.add_argument('--timesteps', type=int, default=5000, help='the length of one evaluation')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    display = Display() if args.show else None
    space = make_world()
    rng = rnd.Random(args.seed)

    g = NEAT.Genome(0, 6, 0, 2, False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0, 1)
    pop = NEAT.Population(g, params, True, 1.0, rng.randint(0, 1000))

    for generation in range(args.generations):
        now = time.time()
        genome_list = NEAT.GetGenomeList(pop)
        for genome in genome_list:
            total_fitness = 0
            for trial in range(args.trials):
                total_fitness += evaluate(genome, space, args.timesteps, rng.randint(80, 400), rng.randint(-200, 200),
                                          rng.randint(80, 400), display)
            genome.SetFitness(total_fitness / args.trials)

        best = max(g.GetFitness() for g in genome_list)
        print('Generation: %d Best fitness: %.1f Species: %d (%.1fs)' % (
            generation, best, len(pop.Species), time.time() - now))

        # evolution is complete if an individual keeps the ball up all the time
        if best >= args.timesteps:
            break

        pop.Epoch()

    if display:
        # Show the best genome's performance forever
        display.fast_mode = False
        display.pygame.display.set_caption("Best genome ever")
        best = pop.GetBestGenome()
        while True:
            evaluate(best, space, 100000, rng.randint(80, 400), rng.randint(-200, 200), rng.randint(80, 400), display)


if __name__ == '__main__':
    main()
