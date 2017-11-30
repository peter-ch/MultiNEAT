#!/usr/bin/python3
import sys
import random as rnd
import cv2
import numpy as np
import MultiNEAT as NEAT
from MultiNEAT.viz import Draw
import pygame
from pygame.locals import *
from pygame.color import *

import pymunk as pm
from pymunk import Vec2d
from pymunk.pygame_util import draw, from_pygame
import progressbar as pbar
import pickle

ns_on = 1

ns_K = 15
ns_recompute_sparseness_each = 20
ns_P_min = 10.0
ns_dynamic_Pmin = True
ns_Pmin_min = 1.0
ns_no_archiving_stagnation_threshold = 150
ns_Pmin_lowering_multiplier = 0.9
ns_Pmin_raising_multiplier = 1.1
ns_quick_archiving_min_evals = 8


max_evaluations = 10000

screen_size_x, screen_size_y = 600, 600
max_timesteps = 1200

collision_type_wall = 0
collision_type_nn = 1
collision_type_ball = 2
collision_type_floor = 3

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
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.75
params.MaxWeight = 8
params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.02
params.MutateRemLinkProb = 0.00

params.Elitism = 0.1
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

rng = NEAT.RNG()
rng.TimeSeed()


class NN_agent:
    def __init__(self, space, brain):
        self.startpos = (480, 80)
        self.radius = 4
        self.mass = 500

        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = collision_type_nn
        self.shape.elasticity = 1.0
        self.body.position = self.startpos

        space.add(self.body, self.shape)

        self.body.velocity_limit = 300

        self.body.velocity = (0, 0)
        self.force = (0,0)

        self.brain = brain
        self.in_air = False

    def touch_floor(self, space, arbiter):
        self.in_air = False
        return True
    def leave_floor(self, space, arbiter):
        self.in_air = True
        return True

    def move(self, x, y):
        self.body.velocity = (x, y)

    def interact(self, ball):
        """
        inputs: x - ball_x, y - ball_y, self_vx, self_vy, 1
        output: x velocity [-1 .. 1]*const, y velocity [-1 .. 1]*const
        """
        inputs = [(self.body.position[0] - ball.body.position[0])/300,
                  (self.body.position[1] - ball.body.position[1])/300,
                  self.body.velocity[0] / 300,
                  self.body.velocity[1] / 300,
                  1.0
                  ]

        self.brain.Input(inputs)
        self.brain.Activate()
        outputs = self.brain.Output()

        self.move(outputs[0] * 200, outputs[1] * 200)



class Ball:
    def __init__(self, space):
        self.mass = 1500
        self.radius = 10
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = collision_type_ball
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        self.body.position = (520, 520)
        space.add(self.body, self.shape)
        self.body.velocity = (0, 0)
        self.body.velocity_limit = 5
        self.in_air = True

class Behavior:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)


def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y+screen_size_y


def evaluate(x):
    gid, genome, space, screen, fast_mode = x
    # Setup the environment
    clock = pygame.time.Clock()

    # The agents - the brain and the ball
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    agent = NN_agent(space, net)
    ball = Ball(space)

    tstep = 0
    bd = 1000000
    while tstep < max_timesteps:
        tstep += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                exit()
            elif event.type == KEYDOWN and event.key == K_f:
                fast_mode = not fast_mode

        ### Update physics
        dt = 1.0/50.0
        space.step(dt)

        # The NN interacts with the world on each 5 timesteps
        if (tstep % 5) == 0:
            agent.interact(ball)

        if not fast_mode:
            # draw the phenotype
            cv2.imshow("current best", Draw(net))
            cv2.waitKey(1)

            ## Draw stuff
            screen.fill(THECOLORS["black"])

            ### Draw stuff
            draw(screen, space)

            ### Flip screen
            pygame.display.flip()
            clock.tick(50)

        d = np.sqrt((ball.body.position[0] - agent.body.position[0])**2 + (ball.body.position[1] - agent.body.position[1])**2)

        if bd > d: bd = d

    fitness = 10000 - bd

    # draw to the screen all genomes ever
    ### Draw stuff
    draw(screen, space)
    ### Flip screen
    pygame.display.flip()

    # remove objects from space
    space.remove(agent.shape, agent.body)
    space.remove(ball.shape, ball.body)

    return fast_mode, gid, fitness, Behavior(agent.body.position[0], agent.body.position[1])


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    if ns_on:
        pygame.display.set_caption("Novelty Search [Press F to turn on/off fast mode]")
    else:
        pygame.display.set_caption("Fitness Search [Press F to turn on/off fast mode]")


    ### Physics stuff
    space = pm.Space()
    space.gravity = Vec2d(0.0, 0.0)

    # walls - the left-top-right walls
    body = pm.Body()
    walls= [# the enclosure
            pm.Segment(body, (50, 50), (50, 550), 5),
            pm.Segment(body, (50, 550), (560, 550), 5),
            pm.Segment(body, (560, 550), (560, 50), 5),
            pm.Segment(body, (50, 50), (560, 50), 5),

            # the obstacle walls
            pm.Segment(body, (120, 480), (560, 480), 5),
            pm.Segment(body, (180, 480), (180, 180), 5),
            pm.Segment(body, (320, 50), (320, 360), 5),
            pm.Segment(body, (440, 480), (440, 360), 5),
            ]

    for s in walls:
        s.friction = 0
        s.elasticity = 0.99
        s.collision_type = collision_type_wall
    space.add(walls)



    g = NEAT.Genome(0, 5, 0, 2, False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)

    best_genome_ever = None
    best_ever = 0
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []

    if not ns_on:

        # rtNEAT mode
        print('============================================================')
        print("Please wait for the initial evaluation to complete.")
        fitnesses = []
        for _, genome in enumerate(NEAT.GetGenomeList(pop)):
            print('Evaluating',_)
            fast_mode, gid, fitness, bh = evaluate((genome.GetID(), genome, space, screen, fast_mode))
            fitnesses.append(fitness)
        for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
            genome.SetFitness(fitness)
            genome.SetEvaluated()
        maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])

        print('======================')
        print('rtNEAT phase')
        pb = pbar.ProgressBar(max_value=max_evaluations)

        for i in range(max_evaluations):
            # get best fitness in population and print it
            fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
            best = max(fitness_list)
            evhist.append(best)
            if best > best_ever:
                sys.stdout.flush()
                print()
                print('NEW RECORD!')
                print('Evaluations:', i, 'Species:', len(pop.Species), 'Fitness:', best)
                best_gs.append(pop.GetBestGenome())
                best_ever = best
                hof.append(pickle.dumps(pop.GetBestGenome()))

            # get the new baby
            old = NEAT.Genome()
            baby = pop.Tick(old)

            # evaluate it
            fast_mode, gid, f, bh = evaluate((baby.GetID(), baby, space, screen, fast_mode))
            baby.SetFitness(f)
            baby.SetEvaluated()

            pb.update(i)
            sys.stdout.flush()
    else:

        archive = []

        # novelty search
        print('============================================================')
        print("Please wait for the initial evaluation to complete.")
        fitnesses = []
        for _, genome in enumerate(NEAT.GetGenomeList(pop)):
            print('Evaluating',_)
            fast_mode, gid, fitness, behavior = evaluate((genome.GetID(), genome, space, screen, fast_mode))
            # associate the behavior with the genome
            genome.behavior = behavior

        # recompute sparseness
        def sparseness(genome):
            distances = []
            for g in NEAT.GetGenomeList(pop):
                d = genome.behavior.distance_to( g.behavior )
                distances.append(d)
            # get the distances from the archive as well
            for ab in archive:
                distances.append( genome.behavior.distance_to(ab) )
            distances = sorted(distances)
            sp = np.mean(distances[1:ns_K+1])
            return sp

        print('======================')
        print('Novelty Search phase')
        pb = pbar.ProgressBar(max_value=max_evaluations)

        # Novelty Search variables
        evaluations = 0
        evals_since_last_archiving = 0
        quick_add_counter = 0

        # initial fitness assignment
        for _, genome in enumerate(NEAT.GetGenomeList(pop)):
            genome.SetFitness( sparseness(genome) )
            genome.SetEvaluated()

        # the Novelty Search tick
        while evaluations < max_evaluations:

            global ns_P_min
            evaluations += 1
            pb.update(evaluations)

            # recompute sparseness for each individual
            if evaluations % ns_recompute_sparseness_each == 0:
                for _, genome in enumerate(NEAT.GetGenomeList(pop)):
                    genome.SetFitness( sparseness(genome) )
                    genome.SetEvaluated()

            # tick
            old = NEAT.Genome()
            new = pop.Tick(old)

            # compute the new behavior
            fast_mode, gid, fitness, behavior = evaluate((new.GetID(), new, space, screen, fast_mode))
            new.behavior = behavior

            # compute sparseness
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
                    ns_P_min *= ns_Pmin_lowering_multiplier
                    if ns_P_min < ns_Pmin_min:
                        ns_P_min = ns_Pmin_min

                # too much additions one after another?
                if quick_add_counter > ns_quick_archiving_min_evals:
                    ns_P_min *= ns_Pmin_raising_multiplier

            # set the fitness of the new individual
            new.SetFitness(sp)
            new.SetEvaluated()

            # still use the objective search's fitness to know which genome is best
            if fitness > best_ever:
                sys.stdout.flush()
                print()
                print('NEW RECORD!')
                print('Evaluations:', evaluations, 'Species:', len(pop.Species), 'Fitness:', fitness)
                hof.append(pickle.dumps(new))
                best_ever = fitness

        pb.finish()

    # Show the best genome's performance forever
    pygame.display.set_caption("Best genome ever")
    while True:
        hg = pickle.loads(hof[-1])
        evaluate((hg.GetID(), hg, space, screen, False))

main()










