#!/usr/bin/python
from __future__ import division
import os
import sys
sys.path.append("/root")
sys.path.append("/home/peter")
sys.path.append("/home/peter/Desktop")
sys.path.append("/home/peter/Desktop/projects")
sys.path.append("/home/peter/Desktop/work")
sys.path.append("/home/peter/code/projects")
sys.path.append("/home/peter/code/work")
sys.path.append("/home/peter/code/common")
import time
import random as rnd
import commands as comm
import cv2
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
import math
import MultiNEAT as NEAT

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk as pm
from pymunk import Vec2d
from pymunk.pygame_util import draw, from_pygame

collision_type_wall = 0
collision_type_nn = 1
collision_type_ball = 2
collision_type_floor = 3

class NN_agent:
    def __init__(self, space, brain, start_x):
        self.startpos = (start_x, 80)
        self.radius = 20
        self.mass = 50000
        
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = collision_type_nn
        self.shape.elasticity = 1.0
        self.body.position = self.startpos
        
        space.add(self.body, self.shape)
        
        self.body.velocity_limit = 1500
        
        self.body.velocity = (230, 0)
        self.force = (0,0)
        
        self.brain = brain
        self.in_air = False
        
    def touch_floor(self, space, arbiter):
        self.in_air = False
        return True
    def leave_floor(self, space, arbiter):
        self.in_air = True
        return True
    
    def jump(self):
        if not self.in_air:
            cur_vel = self.body.velocity
            self.body.velocity = (cur_vel[0], 300)
            
    def move(self, x):
        if not self.in_air:
            #self.body.force = (x, 0)
            self.body.velocity = (x, self.body.velocity[1])
            
    def interact(self, ball):
        """
        inputs: x - ball_x, log(ball_y), log(y), ball_vx, ball_vy, in_air, 1
        output: x velocity [-1 .. 1]*const, jump (if > 0.5 )
        """
        inputs = [(self.body.position[0] - ball.body.position[0])/300, 
#                  math.log(ball.body.position[1]), 
                  math.log(self.body.position[1]),
                  ball.body.velocity[0] / 300, 
                  ball.body.velocity[1] / 300,
                  self.in_air,
                  1.0
                  ]
        
        self.brain.Input(inputs)
        self.brain.Activate()
        outputs = self.brain.Output()
        
        self.move(outputs[0] * 500)
        if outputs[1] > 0.5:
            self.jump()


class Ball:
    def __init__(self, space, start_x, start_vx):
        self.mass = 1500
        self.radius = 30
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = collision_type_ball
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        self.body.position = (start_x, 450)
        space.add(self.body, self.shape)
        self.body.velocity = (start_vx, 0)
        self.body.velocity_limit = 500
        self.in_air = True
        
    def touch_floor(self, space, arbiter):
        self.in_air = False
        return True
    
    def leave_floor(self, space, arbiter):
        self.in_air = True
        return True


screen_size_x, screen_size_y = 600, 600
max_timesteps = 15000

def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y+screen_size_y


def evaluate(genome, space, screen, fast_mode, start_x, start_vx, bot_startx):
    # Setup the environment
    clock = pygame.time.Clock()

    # The agents - the brain and the ball
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    
    agent = NN_agent(space, net, bot_startx)
    ball = Ball(space, start_x, start_vx)
    
    space.add_collision_handler(collision_type_nn,   collision_type_floor, 
                                agent.touch_floor, None, None, agent.leave_floor)
    space.add_collision_handler(collision_type_ball, collision_type_floor, 
                                ball.touch_floor,  None, None, ball.leave_floor)

    tstep = 0
    avg_ball_height = 0
    while tstep < max_timesteps:
        tstep += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                exit()
            elif event.type == KEYDOWN and event.key == K_f:
                fast_mode = not fast_mode
            elif event.type == KEYDOWN and event.key == K_LEFT and not fast_mode:
                ball.body.velocity = (ball.body.velocity[0] - 200, ball.body.velocity[1]) 
            elif event.type == KEYDOWN and event.key == K_RIGHT and not fast_mode:
                ball.body.velocity = (ball.body.velocity[0] + 200, ball.body.velocity[1]) 
            elif event.type == KEYDOWN and event.key == K_UP and not fast_mode:
                ball.body.velocity = (ball.body.velocity[0], ball.body.velocity[1] + 200) 

        ### Update physics
        dt = 1.0/50.0
        space.step(dt)
        
        # The NN interacts with the world on each 20 timesteps
        if (tstep % 20) == 0:
            agent.interact(ball)
        avg_ball_height += ball.body.position[1]
            
        # stopping conditions
        if not ball.in_air:
            break
        #if abs(agent.body.velocity[0]) < 50: # never stop on one place!
        #    break
                    
        if not fast_mode:
            # draw the phenotype
            img = np.zeros((250, 250, 3), dtype=np.uint8)
            img += 10
#            NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
            cv2.imshow("current best", img)
            cv2.waitKey(1)
            
            ## Draw stuff
            screen.fill(THECOLORS["black"])
            
            ### Draw stuff
            draw(screen, space)
            
            ### Flip screen
            pygame.display.flip()
            clock.tick(50)
        
    fitness = tstep #+ avg_ball_height/tstep
    if ball.body.position[1] < 0:
        fitness = 0
    
    # remove objects from space
    space.remove(agent.shape, agent.body)
    space.remove(ball.shape, ball.body)
#    print 'Genome ID:', genome.GetID(), 'Fitness:', tstep
    # the fitness is the number of ticks passed
    return fitness, fast_mode


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("NEAT volleyball")
    

    ### Physics stuff
    space = pm.Space()
    space.gravity = Vec2d(0.0, -500.0)
    
    # walls - the left-top-right walls
    body = pm.Body()
    walls= [pm.Segment(body, (50, 50), (50, 1550), 10)
                ,pm.Segment(body, (50, 1550), (560, 1550), 10)
                ,pm.Segment(body, (560, 1550), (560, 50), 10)
                ]
    
    floor = pm.Segment(body, (50, 50), (560, 50), 10)
    floor.friction = 1.0
    floor.elasticity = 0.0
    floor.collision_type = collision_type_floor
    
    for s in walls:
        s.friction = 0
        s.elasticity = 0.99
        s.collision_type = collision_type_wall
    space.add(walls)
    space.add(floor)
    
    
    
    params = NEAT.Parameters()
    params.PopulationSize = 1000
    params.DynamicCompatibility = True
    params.AllowClones = True
    params.CompatTreshold = 5.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 250
    params.RouletteWheelSelection = True
    params.RecurrentProb = 0.25
    params.OverallMutationRate = 0.33
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    params.MaxWeight = 20
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.05
    params.MutateRemLinkProb = 0.00
    
    rng = NEAT.RNG()
    rng.TimeSeed()
    #rng.Seed(0)
    g = NEAT.Genome(0, 6, 0, 2, False, NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0)
    
    best_genome_ever = None
    fast_mode = False
    for generation in range(1000):
        print "Generation:", generation
        
        now = time.time()
        genome_list = []
        for s in pop.Species:
            for i in s.Individuals:
                genome_list.append(i)
        
        print 'All individuals:', len(genome_list) 
                
        for i, g in enumerate(genome_list):
            total_fitness = 0
            for trial in range(20):
                f, fast_mode = evaluate(g, space, screen, fast_mode, rnd.randint(80, 400), rnd.randint(-200, 200), rnd.randint(80, 400))
                total_fitness += f
            g.SetFitness(total_fitness / 20)
        print 

        best = max([x.GetLeader().GetFitness() for x in pop.Species])
        print 'Best fitness:', best, 'Species:', len(pop.Species)

        
        # Draw the best genome's phenotype
        net = NEAT.NeuralNetwork()
        best_genome_ever = pop.Species[0].GetLeader()
        best_genome_ever.BuildPhenotype(net)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
        cv2.imshow("current best", img)
        cv2.waitKey(1)
        
        if best >= 10000:
            break # evolution is complete if an individual keeps the ball up for that many timesteps
        #if pop.GetStagnation() > 500:
        #    break
        
        print "Evaluation took", time.time() - now, "seconds."
        print "Reproducing.."
        now = time.time()
        pop.Epoch()
        print "Reproduction took", time.time() - now, "seconds."
        
    # Show the best genome's performance forever
    pygame.display.set_caption("Best genome ever")
    while True:
        evaluate(best_genome_ever, space, screen, False, rnd.randint(80, 400), rnd.randint(-200, 200), rnd.randint(80, 400))

main()










