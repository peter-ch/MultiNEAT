"""A small 2D physics engine for simulated tasks (implemented in C++).

It is meant for evaluating agents in simple worlds - balls, walls, floors - and
has an API similar to a subset of pymunk::

    from MultiNEAT import physics2d as phys

    space = phys.Space()
    space.gravity = (0, -500)

    walls = phys.Body.static()
    floor = phys.Segment(walls, (0, 0), (600, 0), 5)
    floor.elasticity = 0.5

    ball_body = phys.Body(mass=1.0)
    ball_body.position = (300, 400)
    ball = phys.Circle(ball_body, 10)
    ball.elasticity = 0.9

    space.add(walls, floor, ball_body, ball)
    for _ in range(100):
        space.step(1 / 50)

Bodies don't rotate. Shapes are circles and thick line segments (capsules).
Collision callbacks are registered per pair of collision types with
``space.add_collision_handler(type_a, type_b, begin=f, separate=g)`` and are
called as ``f(shape_a, shape_b)``.
"""
from . import _MultiNEAT

_p = _MultiNEAT._physics2d

Vec2d = _p.Vec2d
Body = _p.Body
Shape = _p.Shape
Circle = _p.Circle
Segment = _p.Segment
Space = _p.Space

__all__ = ['Vec2d', 'Body', 'Shape', 'Circle', 'Segment', 'Space', 'draw_pygame']


def draw_pygame(surface, space, flip_y=True, colors=None):
    """Draws all shapes of the space on a pygame surface.

    With flip_y the y axis points up (y=0 is the bottom of the surface).
    colors can map collision types to RGB tuples.
    """
    import pygame

    height = surface.get_height()

    def pt(v):
        return int(v[0]), int(height - v[1] if flip_y else v[1])

    colors = colors or {}
    for shape in space.shapes:
        color = colors.get(shape.collision_type)
        if shape.is_circle:
            color = color or ((120, 120, 255) if not shape.body.is_static else (200, 200, 200))
            r = max(int(shape.radius), 1)
            pygame.draw.circle(surface, color, pt(shape.world_center), r)
        else:
            color = color or (200, 200, 200)
            width = max(int(shape.radius * 2), 1)
            pygame.draw.line(surface, color, pt(shape.world_a), pt(shape.world_b), width)
