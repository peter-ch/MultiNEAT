#ifndef MULTINEAT_PHYSICS2D_H
#define MULTINEAT_PHYSICS2D_H

///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Physics2D.h
// Description: A small 2D rigid body physics engine for evaluating agents in
//              simple simulated worlds (used by the examples).
//
// Features:
//   - dynamic and static bodies (no rotation - all shapes are rotation invariant)
//   - circle shapes and thick line segment shapes (capsules)
//   - gravity, external forces, per-body speed limit
//   - impulse based collision response with restitution (elasticity) and friction
//   - collision callbacks when two shapes begin and stop touching, filtered by
//     collision type (like pymunk/Chipmunk)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NEAT
{
namespace Physics2D
{

struct Vec2
{
    double x, y;

    Vec2() : x(0), y(0) {}
    Vec2(double a_x, double a_y) : x(a_x), y(a_y) {}

    Vec2 operator+(const Vec2 &o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2 &o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator-() const { return Vec2(-x, -y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    Vec2 operator/(double s) const { return Vec2(x / s, y / s); }
    Vec2 &operator+=(const Vec2 &o) { x += o.x; y += o.y; return *this; }
    Vec2 &operator-=(const Vec2 &o) { x -= o.x; y -= o.y; return *this; }
    bool operator==(const Vec2 &o) const { return (x == o.x) && (y == o.y); }

    double Dot(const Vec2 &o) const { return x * o.x + y * o.y; }
    double Length() const { return std::sqrt(x * x + y * y); }
    double LengthSq() const { return x * x + y * y; }
};

class Body
{
public:
    enum Type { DYNAMIC = 0, STATIC = 1 };

    Type type;
    double mass;
    Vec2 position;
    Vec2 velocity;
    Vec2 force;            // applied every step until changed
    double velocity_limit; // maximum speed, 0 means no limit

    // a dynamic body with the given mass
    explicit Body(double a_mass = 1.0);

    // a static (immovable) body
    static std::shared_ptr<Body> Static();

    double InverseMass() const { return ((type == STATIC) || (mass <= 0)) ? 0.0 : 1.0 / mass; }

    void ApplyImpulse(const Vec2 &a_impulse) { velocity += a_impulse * InverseMass(); }
};

class Shape
{
public:
    enum Kind { CIRCLE = 0, SEGMENT = 1 };

    Kind kind;
    std::shared_ptr<Body> body;
    double radius;          // circle radius / half thickness of a segment
    Vec2 offset;            // circle center relative to the body
    Vec2 a, b;              // segment end points relative to the body
    double elasticity;      // 0 .. 1
    double friction;        // >= 0
    int collision_type;     // used to select collision callbacks
    bool sensor;            // sensors report collisions but don't collide

    Shape(Kind a_kind, std::shared_ptr<Body> a_body, double a_radius);

    Vec2 WorldCenter() const { return body->position + offset; }
    Vec2 WorldA() const { return body->position + a; }
    Vec2 WorldB() const { return body->position + b; }
};

// convenience constructors
std::shared_ptr<Shape> MakeCircle(std::shared_ptr<Body> a_body, double a_radius, Vec2 a_offset = Vec2());
std::shared_ptr<Shape> MakeSegment(std::shared_ptr<Body> a_body, Vec2 a_a, Vec2 a_b, double a_radius);

// Called with the two shapes, in the order the handler was registered with
typedef std::function<void(std::shared_ptr<Shape>, std::shared_ptr<Shape>)> CollisionCallback;

class Space
{
public:
    Vec2 gravity;
    int iterations;          // collision solver iterations per step
    double slop;             // allowed penetration before positions are corrected
    double correction;       // fraction of the penetration corrected per iteration

    Space();

    void AddBody(std::shared_ptr<Body> a_body);
    void AddShape(std::shared_ptr<Shape> a_shape);
    void RemoveBody(std::shared_ptr<Body> a_body);
    void RemoveShape(std::shared_ptr<Shape> a_shape); // also stops the contacts of the shape

    // callbacks for shapes with these collision types starting/stopping to touch
    void AddCollisionHandler(int a_type_a, int a_type_b, CollisionCallback a_begin, CollisionCallback a_separate);
    void RemoveCollisionHandler(int a_type_a, int a_type_b);

    // advances the simulation by a_dt seconds
    void Step(double a_dt);

    // true if the two shapes are touching (as of the last step)
    bool Touching(const std::shared_ptr<Shape> &a_s1, const std::shared_ptr<Shape> &a_s2) const;

    const std::vector< std::shared_ptr<Body> > &Bodies() const { return m_bodies; }
    const std::vector< std::shared_ptr<Shape> > &Shapes() const { return m_shapes; }

private:
    struct Contact
    {
        Shape *s1, *s2;
        Vec2 normal;        // from s1 to s2
        double penetration;
    };

    struct Handler
    {
        CollisionCallback begin, separate;
    };

    std::vector< std::shared_ptr<Body> > m_bodies;
    std::vector< std::shared_ptr<Shape> > m_shapes;
    std::map< std::pair<int, int>, Handler > m_handlers;
    std::set< std::pair<Shape *, Shape *> > m_touching;

    bool Collide(Shape *a_s1, Shape *a_s2, Contact &a_contact) const;
    void Resolve(const Contact &a_c, double a_dt);
    std::shared_ptr<Shape> Find(Shape *a_s) const;
    void Notify(Shape *a_s1, Shape *a_s2, bool a_begin);
};

} // namespace Physics2D
} // namespace NEAT

#endif // MULTINEAT_PHYSICS2D_H
