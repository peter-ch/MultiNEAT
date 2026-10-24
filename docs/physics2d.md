# Physics engine

`MultiNEAT.physics2d` is a small 2D rigid body engine, written in C++, for evaluating agents in simple
simulated worlds - robots in a maze, balls, walls and floors. Its API resembles a subset of pymunk.

```python
from MultiNEAT import physics2d as phys

space = phys.Space()
space.gravity = (0, -500)

# static geometry
walls = phys.Body.static()
floor = phys.Segment(walls, (0, 0), (600, 0), 5)   # from, to, half thickness
floor.elasticity = 0.5
floor.friction = 1.0

# a ball
body = phys.Body(mass=1.0)
body.position = (300, 400)
ball = phys.Circle(body, 10)
ball.elasticity = 0.9

space.add(walls, floor, body, ball)

for step in range(500):
    space.step(1 / 50)
print(body.position, body.velocity)
```

## Bodies

`Body(mass)` creates a dynamic body, `Body.static()` an immovable one (walls, floors). Bodies don't
rotate, so all shapes behave like circles and capsules.

| Attribute | |
|---|---|
| `position`, `velocity` | `Vec2d` - tuples and lists are accepted when setting |
| `force` | a constant force applied every step until changed |
| `mass` | |
| `velocity_limit` | the maximum speed, 0 = unlimited |
| `apply_impulse(v)` | changes the velocity by `v / mass` |
| `is_static` | |

## Shapes

`Circle(body, radius, offset=(0, 0))` and `Segment(body, a, b, radius)` create shapes attached to a
body. Segments are line segments with thickness - capsules - and are meant for static geometry (segments
don't collide with each other).

| Attribute | |
|---|---|
| `elasticity` | 0..1, the restitution of a collision is the product of both shapes' values |
| `friction` | the friction coefficient is the product of both shapes' values |
| `collision_type` | an integer used to select collision callbacks |
| `sensor` | sensors report contacts but don't collide |
| `radius`, `offset`, `a`, `b` | the geometry, relative to the body |
| `world_center`, `world_a`, `world_b` | the geometry in world coordinates |

## The space

| | |
|---|---|
| `add(*objects)`, `remove(*objects)` | add/remove bodies and shapes, or lists of them |
| `step(dt)` | advance the simulation by `dt` seconds |
| `gravity` | |
| `iterations` | collision solver iterations per step (default 8) |
| `bodies`, `shapes` | lists of the objects in the space |
| `touching(shape_a, shape_b)` | whether two shapes touched in the last step |
| `add_collision_handler(type_a, type_b, begin=None, separate=None)` | `begin(a, b)` is called when shapes with these collision types start touching, `separate(a, b)` when they stop |
| `remove_collision_handler(type_a, type_b)` | |

```python
ROBOT, FLOOR = 1, 2
robot.collision_type = ROBOT
floor.collision_type = FLOOR

state = {'on_floor': False}
space.add_collision_handler(ROBOT, FLOOR,
                            begin=lambda a, b: state.update(on_floor=True),
                            separate=lambda a, b: state.update(on_floor=False))
```

The engine uses semi-implicit Euler integration and impulse based collision resolution with
restitution, Coulomb friction and positional correction. Slow contacts don't bounce, so objects come
to rest. Keep the per-step movement smaller than the thickness of the walls (speed x dt < radius) to
avoid tunneling.

## Drawing

`phys.draw_pygame(surface, space, flip_y=True, colors=None)` draws all shapes on a pygame surface
(y points up when `flip_y` is set). `colors` maps collision types to RGB tuples.

See `examples/NoveltySearch.py` and `examples/ball_keeper.py`.
