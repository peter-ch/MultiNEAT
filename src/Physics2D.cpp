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
// File:        Physics2D.cpp
// Description: A small 2D rigid body physics engine. See Physics2D.h
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <stdexcept>

#include "Physics2D.h"

namespace NEAT
{
namespace Physics2D
{

Body::Body(double a_mass)
    : type(DYNAMIC), mass(a_mass), velocity_limit(0)
{
    if (!(a_mass > 0))
    {
        throw std::invalid_argument("A dynamic body needs a positive mass (use Body.static() for immovable bodies)");
    }
}

std::shared_ptr<Body> Body::Static()
{
    std::shared_ptr<Body> b = std::make_shared<Body>(1.0);
    b->type = STATIC;
    b->mass = 0;
    return b;
}

Shape::Shape(Kind a_kind, std::shared_ptr<Body> a_body, double a_radius)
    : kind(a_kind), body(a_body), radius(a_radius), elasticity(0), friction(0), collision_type(0), sensor(false)
{
    if (!body)
    {
        throw std::invalid_argument("A shape needs a body");
    }
    if (a_radius < 0)
    {
        throw std::invalid_argument("The radius can't be negative");
    }
}

std::shared_ptr<Shape> MakeCircle(std::shared_ptr<Body> a_body, double a_radius, Vec2 a_offset)
{
    std::shared_ptr<Shape> s = std::make_shared<Shape>(Shape::CIRCLE, a_body, a_radius);
    s->offset = a_offset;
    return s;
}

std::shared_ptr<Shape> MakeSegment(std::shared_ptr<Body> a_body, Vec2 a_a, Vec2 a_b, double a_radius)
{
    std::shared_ptr<Shape> s = std::make_shared<Shape>(Shape::SEGMENT, a_body, a_radius);
    s->a = a_a;
    s->b = a_b;
    return s;
}


Space::Space()
    : iterations(8), slop(0.05), correction(0.8)
{
}

void Space::AddBody(std::shared_ptr<Body> a_body)
{
    if (std::find(m_bodies.begin(), m_bodies.end(), a_body) == m_bodies.end())
    {
        m_bodies.push_back(a_body);
    }
}

void Space::AddShape(std::shared_ptr<Shape> a_shape)
{
    if (std::find(m_shapes.begin(), m_shapes.end(), a_shape) == m_shapes.end())
    {
        m_shapes.push_back(a_shape);
    }
}

void Space::RemoveBody(std::shared_ptr<Body> a_body)
{
    m_bodies.erase(std::remove(m_bodies.begin(), m_bodies.end(), a_body), m_bodies.end());
}

void Space::RemoveShape(std::shared_ptr<Shape> a_shape)
{
    Shape *t_s = a_shape.get();
    // the shape stops touching everything, but no callbacks are called for that
    for (auto it = m_touching.begin(); it != m_touching.end();)
    {
        if ((it->first == t_s) || (it->second == t_s))
        {
            it = m_touching.erase(it);
        }
        else
        {
            ++it;
        }
    }
    m_shapes.erase(std::remove(m_shapes.begin(), m_shapes.end(), a_shape), m_shapes.end());
}

void Space::AddCollisionHandler(int a_type_a, int a_type_b, CollisionCallback a_begin, CollisionCallback a_separate)
{
    Handler h;
    h.begin = a_begin;
    h.separate = a_separate;
    m_handlers[std::make_pair(a_type_a, a_type_b)] = h;
}

void Space::RemoveCollisionHandler(int a_type_a, int a_type_b)
{
    m_handlers.erase(std::make_pair(a_type_a, a_type_b));
}

static Vec2 ClosestPointOnSegment(const Vec2 &a_p, const Vec2 &a_a, const Vec2 &a_b)
{
    Vec2 ab = a_b - a_a;
    double len_sq = ab.LengthSq();
    if (len_sq == 0)
    {
        return a_a;
    }
    double t = (a_p - a_a).Dot(ab) / len_sq;
    t = std::max(0.0, std::min(1.0, t));
    return a_a + ab * t;
}

// Fills the contact if the shapes overlap (or touch within the slop)
bool Space::Collide(Shape *a_s1, Shape *a_s2, Contact &a_contact) const
{
    if ((a_s1->kind == Shape::SEGMENT) && (a_s2->kind == Shape::SEGMENT))
    {
        return false; // not supported - segments are meant to be static walls
    }

    // make sure s1 is the segment if there is one
    bool t_swapped = false;
    if ((a_s2->kind == Shape::SEGMENT) && (a_s1->kind == Shape::CIRCLE))
    {
        std::swap(a_s1, a_s2);
        t_swapped = true;
    }

    Vec2 t_from, t_to;
    if (a_s1->kind == Shape::SEGMENT)
    {
        t_to = a_s2->WorldCenter();
        t_from = ClosestPointOnSegment(t_to, a_s1->WorldA(), a_s1->WorldB());
    }
    else
    {
        t_from = a_s1->WorldCenter();
        t_to = a_s2->WorldCenter();
    }

    Vec2 d = t_to - t_from;
    double dist = d.Length();
    double pen = (a_s1->radius + a_s2->radius) - dist;
    if (pen < -slop)
    {
        return false;
    }

    Vec2 n;
    if (dist > 1e-12)
    {
        n = d / dist;
    }
    else if (a_s1->kind == Shape::SEGMENT)
    {
        Vec2 ab = a_s1->WorldB() - a_s1->WorldA();
        n = Vec2(-ab.y, ab.x);
        double l = n.Length();
        n = (l > 0) ? n / l : Vec2(0, 1);
    }
    else
    {
        n = Vec2(0, 1);
    }

    a_contact.s1 = t_swapped ? a_s2 : a_s1;
    a_contact.s2 = t_swapped ? a_s1 : a_s2;
    a_contact.normal = t_swapped ? -n : n;
    a_contact.penetration = pen;
    return true;
}

void Space::Resolve(const Contact &a_c, double a_dt)
{
    Body *A = a_c.s1->body.get();
    Body *B = a_c.s2->body.get();
    double inv_a = A->InverseMass();
    double inv_b = B->InverseMass();
    double inv_sum = inv_a + inv_b;
    if (inv_sum == 0)
    {
        return;
    }

    const Vec2 &n = a_c.normal;

    // velocity
    Vec2 rv = B->velocity - A->velocity;
    double vn = rv.Dot(n);
    if (vn < 0)
    {
        // no bouncing for slow (resting) contacts, so objects can come to rest
        double e = a_c.s1->elasticity * a_c.s2->elasticity;
        if (-vn < 2.0 * gravity.Length() * a_dt)
        {
            e = 0;
        }
        double j = -(1.0 + e) * vn / inv_sum;
        Vec2 impulse = n * j;
        A->velocity -= impulse * inv_a;
        B->velocity += impulse * inv_b;

        // friction
        double mu = a_c.s1->friction * a_c.s2->friction;
        if (mu > 0)
        {
            rv = B->velocity - A->velocity;
            Vec2 t = rv - n * rv.Dot(n);
            double t_len = t.Length();
            if (t_len > 1e-12)
            {
                t = t / t_len;
                double jt = -rv.Dot(t) / inv_sum;
                double max_jt = mu * j;
                jt = std::max(-max_jt, std::min(max_jt, jt));
                Vec2 f_impulse = t * jt;
                A->velocity -= f_impulse * inv_a;
                B->velocity += f_impulse * inv_b;
            }
        }
    }

    // position correction
    double pen = a_c.penetration - slop;
    if (pen > 0)
    {
        Vec2 corr = n * (correction * pen / inv_sum);
        A->position -= corr * inv_a;
        B->position += corr * inv_b;
    }
}

std::shared_ptr<Shape> Space::Find(Shape *a_s) const
{
    for (const auto &s : m_shapes)
    {
        if (s.get() == a_s)
        {
            return s;
        }
    }
    return std::shared_ptr<Shape>();
}

void Space::Notify(Shape *a_s1, Shape *a_s2, bool a_begin)
{
    auto it = m_handlers.find(std::make_pair(a_s1->collision_type, a_s2->collision_type));
    bool t_reversed = false;
    if (it == m_handlers.end())
    {
        it = m_handlers.find(std::make_pair(a_s2->collision_type, a_s1->collision_type));
        t_reversed = true;
    }
    if (it == m_handlers.end())
    {
        return;
    }
    // copy - the callback may modify the handlers
    CollisionCallback cb = a_begin ? it->second.begin : it->second.separate;
    if (!cb)
    {
        return;
    }
    std::shared_ptr<Shape> p1 = Find(a_s1), p2 = Find(a_s2);
    if (!p1 || !p2)
    {
        return;
    }
    if (t_reversed)
    {
        cb(p2, p1);
    }
    else
    {
        cb(p1, p2);
    }
}

void Space::Step(double a_dt)
{
    if (!(a_dt > 0))
    {
        return;
    }

    // integrate velocities and positions (semi-implicit Euler)
    for (auto &b : m_bodies)
    {
        if (b->type == Body::STATIC)
        {
            continue;
        }
        b->velocity += (gravity + b->force * b->InverseMass()) * a_dt;
        if (b->velocity_limit > 0)
        {
            double speed = b->velocity.Length();
            if (speed > b->velocity_limit)
            {
                b->velocity = b->velocity * (b->velocity_limit / speed);
            }
        }
        b->position += b->velocity * a_dt;
    }

    // collisions
    std::set< std::pair<Shape *, Shape *> > t_touching;
    for (int it = 0; it < std::max(1, iterations); it++)
    {
        for (size_t i = 0; i < m_shapes.size(); i++)
        {
            Shape *s1 = m_shapes[i].get();
            for (size_t j = i + 1; j < m_shapes.size(); j++)
            {
                Shape *s2 = m_shapes[j].get();
                if (s1->body == s2->body)
                {
                    continue;
                }
                if ((s1->body->type == Body::STATIC) && (s2->body->type == Body::STATIC))
                {
                    continue;
                }

                Contact c;
                if (Collide(s1, s2, c))
                {
                    if (it == 0)
                    {
                        t_touching.insert(std::make_pair(s1, s2));
                    }
                    if (!s1->sensor && !s2->sensor)
                    {
                        Resolve(c, a_dt);
                    }
                }
            }
        }
    }

    // the speed limit also applies after collisions
    for (auto &b : m_bodies)
    {
        if ((b->type != Body::STATIC) && (b->velocity_limit > 0))
        {
            double speed = b->velocity.Length();
            if (speed > b->velocity_limit)
            {
                b->velocity = b->velocity * (b->velocity_limit / speed);
            }
        }
    }

    // callbacks
    std::set< std::pair<Shape *, Shape *> > t_old = m_touching;
    m_touching = t_touching;
    for (const auto &p : t_touching)
    {
        if (t_old.count(p) == 0)
        {
            Notify(p.first, p.second, true);
        }
    }
    for (const auto &p : t_old)
    {
        if (t_touching.count(p) == 0)
        {
            Notify(p.first, p.second, false);
        }
    }
}

bool Space::Touching(const std::shared_ptr<Shape> &a_s1, const std::shared_ptr<Shape> &a_s2) const
{
    return (m_touching.count(std::make_pair(a_s1.get(), a_s2.get())) > 0) ||
           (m_touching.count(std::make_pair(a_s2.get(), a_s1.get())) > 0);
}

} // namespace Physics2D
} // namespace NEAT
