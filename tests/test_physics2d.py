"""Tests for the built-in 2D physics engine (MultiNEAT.physics2d)."""
import pickle

import pytest

from MultiNEAT import physics2d as phys


def make_world(elasticity=0.0, gravity=(0, -500)):
    space = phys.Space()
    space.gravity = gravity
    static = phys.Body.static()
    floor = phys.Segment(static, (0, 0), (1000, 0), 5)
    floor.elasticity = 1.0
    floor.collision_type = 1
    left = phys.Segment(static, (0, 0), (0, 1000), 5)
    right = phys.Segment(static, (1000, 0), (1000, 1000), 5)
    for w in (left, right):
        w.elasticity = 1.0
    space.add(static, [floor, left, right])
    body = phys.Body(1.0)
    body.position = (500, 300)
    ball = phys.Circle(body, 10)
    ball.elasticity = elasticity
    ball.collision_type = 2
    space.add(body, ball)
    return space, body, ball, floor


def test_vec2d():
    v = phys.Vec2d(3, 4)
    assert v.length() == 5
    assert tuple(v) == (3, 4) and v[0] == 3 and v[1] == 4 and len(v) == 2
    assert (v + phys.Vec2d(1, 1)) == phys.Vec2d(4, 5)
    assert pickle.loads(pickle.dumps(v)) == v
    b = phys.Body(2.0)
    b.position = (1, 2)  # tuples convert implicitly
    assert tuple(b.position) == (1, 2)


def test_ball_comes_to_rest_on_the_floor():
    space, body, ball, floor = make_world(elasticity=0.0)
    for _ in range(300):
        space.step(1 / 50)
    # resting on the floor: center at floor thickness + radius
    assert body.position.y == pytest.approx(15, abs=0.5)
    assert abs(body.velocity.y) < 15
    assert space.touching(ball, floor)


def test_bounce_keeps_energy_with_full_elasticity():
    space, body, ball, floor = make_world(elasticity=1.0)
    heights = []
    going_up = False
    for _ in range(1500):
        space.step(1 / 200)
        if body.velocity.y > 0:
            going_up = True
        elif going_up and body.velocity.y <= 0:
            heights.append(body.position.y)
            going_up = False
    assert len(heights) >= 2
    assert heights[-1] > 0.85 * 300


def test_collision_callbacks():
    space, body, ball, floor = make_world(elasticity=0.5)
    events = []
    space.add_collision_handler(2, 1, begin=lambda a, b: events.append(('begin', a.collision_type, b.collision_type)),
                                separate=lambda a, b: events.append(('separate', a.collision_type, b.collision_type)))
    for _ in range(400):
        space.step(1 / 50)
    assert events[0] == ('begin', 2, 1)
    assert ('separate', 2, 1) in events  # it bounced


def test_circle_collision_conserves_momentum():
    space = phys.Space()
    a = phys.Body(1.0)
    b = phys.Body(3.0)
    a.position = (0, 0)
    b.position = (50, 0)
    a.velocity = (100, 0)
    ca, cb = phys.Circle(a, 10), phys.Circle(b, 10)
    ca.elasticity = cb.elasticity = 1.0
    space.add(a, b, ca, cb)
    p0 = a.mass * a.velocity.x + b.mass * b.velocity.x
    for _ in range(100):
        space.step(1 / 100)
    p1 = a.mass * a.velocity.x + b.mass * b.velocity.x
    assert p1 == pytest.approx(p0, rel=1e-6)
    assert b.velocity.x > 0 and a.velocity.x < 0  # elastic bounce off the heavier ball


def test_velocity_limit_and_no_tunneling():
    space, body, ball, floor = make_world(elasticity=1.0, gravity=(0, 0))
    body.velocity_limit = 300
    body.velocity = (5000, 0)
    for _ in range(1000):
        space.step(1 / 50)
        assert body.velocity.length() <= 300 + 1e-9
        assert 0 < body.position.x < 1000


def test_remove_and_sensor():
    space, body, ball, floor = make_world()
    ball.sensor = True
    hits = []
    space.add_collision_handler(2, 1, begin=lambda a, b: hits.append(1))
    for _ in range(100):
        space.step(1 / 50)
    assert hits  # sensors report contacts...
    assert body.position.y < 0  # ...but don't collide
    space.remove(ball, body)
    assert len(space.shapes) == 3 and len(space.bodies) == 1
