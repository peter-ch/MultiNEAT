//****************************************************************************
//
// File         : vector.h
//
// Description  : Misc 2D/3D vector structures.
//
//
//****************************************************************************

#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>


#define VERY_SMALL 0.000001



inline double sqr(double x)
{
    return x*x;
}


class vector2D
{
public:

    double  x, y;

    double magnitude()
    {
        return sqrt( sqr(x) + sqr(y) );
    }

    double sqr_magnitude()
    {
        return ( sqr(x) + sqr(y) );
    }

    void normalize()
    {
        double mag = magnitude();

        if (mag != 0)
        {
            x /= mag;
            y /= mag;
        }
    }

    // Initialize
    vector2D()
    {
        x = 0;
        y = 0;
    }

    vector2D(double xx, double yy)
    {
        x = xx;
        y = yy;
    }

    // Make it point from p1 to p2
    vector2D(vector2D& p1, vector2D& p2)
    {
        x = p2.x - p1.x;
        y = p2.y - p1.y;
    }


    vector2D plus(vector2D& vec)
    {
        vector2D tmp;

        tmp.x = x + vec.x;
        tmp.y = y + vec.y;

        return tmp;
    }

    vector2D minus(vector2D& vec)
    {
        vector2D tmp;

        tmp.x = x - vec.x;
        tmp.y = y - vec.y;

        return tmp;
    }

    void add(vector2D& vec)
    {
        x += vec.x;
        y += vec.y;
    }
    void sub(vector2D& vec)
    {
        x -= vec.x;
        y -= vec.y;
    }

    double distance_to(vector2D& vec)
    {
        return sqrt( sqr(vec.x - x) + sqr(vec.y - y) );
    }

    double sqr_distance_to(vector2D& vec)
    {
        return ( sqr(vec.x - x) + sqr(vec.y - y) );
    }

    double dot(vector2D& vec)
    {
        return ((x * vec.x) + (y * vec.y));
    }

    bool inside_circle(vector2D& c, double r)
    {
        if  ((sqr(x - c.x) + sqr(y - c.y))  < sqr(r))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void negate()
    {
        x = -x;
        y = -y;
    }
};


class vector3D
{
public:

    double  x, y, z;

    double magnitude()
    {
        return sqrt( sqr(x) + sqr(y) + sqr(z));
    }

    double sqr_magnitude()
    {
        return ( sqr(x) + sqr(y) + sqr(z));
    }

    void normalize()
    {
        double mag = magnitude();

        if (mag != 0)
        {
            x /= mag;
            y /= mag;
            z /= mag;
        }
    }

    // Initialize
    vector3D()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    vector3D(double xx, double yy, double zz)
    {
        x = xx;
        y = yy;
        z = zz;
    }

    // Make it point from p1 to p2
    vector3D(vector3D& p1, vector3D& p2)
    {
        x = p2.x - p1.x;
        y = p2.y - p1.y;
        z = p2.z - p1.z;
    }


    vector3D plus(vector3D& vec)
    {
        vector3D tmp;

        tmp.x = x + vec.x;
        tmp.y = y + vec.y;
        tmp.z = z + vec.z;

        return tmp;
    }

    vector3D minus(vector3D& vec)
    {
        vector3D tmp;

        tmp.x = x - vec.x;
        tmp.y = y - vec.y;
        tmp.z = z - vec.z;

        return tmp;
    }

    void add(vector3D& vec)
    {
        x += vec.x;
        y += vec.y;
        z += vec.z;
    }
    void sub(vector3D& vec)
    {
        x -= vec.x;
        y -= vec.y;
        z -= vec.z;
    }

    double distance_to(vector3D& vec)
    {
        return sqrt( sqr(vec.x - x) + sqr(vec.y - y) + sqr(vec.z - z) );
    }

    double sqr_distance_to(vector3D& vec)
    {
        return ( sqr(vec.x - x) + sqr(vec.y - y) + sqr(vec.z - z) );
    }

    double dot(vector3D& vec)
    {
        return ((x * vec.x) + (y * vec.y) + (z * vec.z));
    }

    bool inside_sphere(vector3D& c, double r)
    {
        if  ((sqr(x - c.x) + sqr(y - c.y) + sqr(z - c.z))  < sqr(r))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void negate()
    {
        x = -x;
        y = -y;
        z = -z;
    }
};

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))

class line2D
{
public:

    vector2D p1, p2;

//  Initialize from two points
    line2D(vector2D& i1, vector2D& i2)
    {
        p1 = i1;
        p2 = i2;
    }

    line2D()
    {
        p1.x = 0;
        p1.y = 0;
        p2.x = 0;
        p2.y = 0;
    }


//  Determine at which side of a line the point x,y lies ------------------
    int side_of(vector2D& v)
    {
        double s = (p1.y - v.y)*(p2.x - p1.x) - (p1.x - v.x)*(p2.y - p1.y);
        if (s < 0) return -1;
        else if (s > 0) return 1;
        else return 0;
    }

    bool intersect_circle(vector2D& Center, double radius, vector2D& isect)
    {

        double t=0,t1=0,t2=0;
        double disc=0;

        vector2D delta, P(p1.x, p1.y), D(p1, p2); // D = p1 -> p2

        D.normalize();

        delta = Center;
        delta.sub(P); // delta = P - Center

        disc = sqr(D.dot(delta)) - (D.sqr_magnitude()) * ((delta.sqr_magnitude()) - sqr(radius));

        if      (disc < 0) return false;
        else if (disc == 0)
        {
            t = -(D.dot(delta)) / (D.sqr_magnitude());

            if (t>0) return false;

            isect = p1;

            isect.x += t*D.x;
            isect.y += t*D.y;

            if (isect.sqr_distance_to(p1) > p2.sqr_distance_to(p1)) return false;
            else return true;
        }
        else
        {
            t1 = (-(D.dot(delta)) + sqrt(disc)) / (D.sqr_magnitude());
            t2 = (-(D.dot(delta)) - sqrt(disc)) / (D.sqr_magnitude());

            t = MAX(t1,t2);

//			if (t>0) return false;

            isect = p1;

            isect.x += t*D.x;
            isect.y += t*D.y;

            if (isect.sqr_distance_to(p1) > p2.sqr_distance_to(p1)) return false;
            else return true;
        }

        return true;
    }

};


//--------------------2LinesIntersection2D-------------------------
//
//	Given 2 lines in 2D space AB, CD this returns true if an
//	intersection occurs and sets dist to the distance the intersection
//  occurs along AB
//
//-----------------------------------------------------------------
inline bool LineIntersection2D(vector2D A,
                               vector2D B,
                               vector2D C,
                               vector2D D,
                               double &dist)
{
    //first test against the bounding boxes of the lines
    if ( (((A.y > D.y) && (B.y > D.y)) && ((A.y > C.y) && (B.y > C.y))) ||
            (((B.y < C.y) && (A.y < C.y)) && ((B.y < D.y) && (A.y < D.y))) ||
            (((A.x > D.x) && (B.x > D.x)) && ((A.x > C.x) && (B.x > C.x))) ||
            (((B.x < C.x) && (A.x < C.x)) && ((B.x < D.x) && (A.x < D.x))) )
    {
        dist = 0;

        return false;
    }

    double rTop = (A.y-C.y)*(D.x-C.x)-(A.x-C.x)*(D.y-C.y);
    double rBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);

    double sTop = (A.y-C.y)*(B.x-A.x)-(A.x-C.x)*(B.y-A.y);
    double sBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);


    double rTopBot = rTop*rBot;
    double sTopBot = sTop*sBot;

    if ((rTopBot>0) && (rTopBot<rBot*rBot) && (sTopBot>0) && (sTopBot<sBot*sBot))
    {

        dist = rTop/rBot;

        return true;
    }


    else
    {
        dist = 0;

        return false;
    }

}


/*
typedef class
{
public:

   double       x, y, z;
} Vertice3D, Vector3D, Vertex3D, Point3D, vertice3D, vector3D, vertex3D, point3D;

*/


/*
typedef class
{
public:

   vector3D normal; // Normal Vector
   double d;         // Intercept
} Plane, plane;


typedef class
{
public:

   int   Num_Planes;
   plane Clipping_Planes[MAX_FRUSTUM_CLIPPING_PLANES];

} Frustum;





// ********************************* //
// INLINE VECTOR OPERATION FUNCTIONS //
// ********************************* //


inline void Vector3D_Add(Vector3D *d, Vector3D *s)
{
    d->x += s->x;
    d->y += s->y;
    d->z += s->z;
}

inline void Vector3D_Sub(Vector3D *d, Vector3D *s)
{
    d->x -= s->x;
    d->y -= s->y;
    d->z -= s->z;
}


//--- Make vector from two vertices -----------------------------------------
// The Vector Made points from v2 to v1
inline void Make_Vector3D(Vertex3D *v1, Vertex3D *v2, Vector3D *u)
{
    u->x = v1->x - v2->x;
    u->y = v1->y - v2->y;
    u->z = v1->z - v2->z;
}

inline void Make_Vector2D(Vertex2D *v1, Vertex2D *v2, Vector2D *u)
{
    u->x = v1->x - v2->x;
    u->y = v1->y - v2->y;
}


//--- Calculate lenght of a vector ------------------------------------------
inline double Vector3D_Magnitude(Vector3D *u)
{
    return(sqrt(u->x * u->x + u->y * u->y + u->z * u->z));
}
//--- Calculate lenght of a vector ------------------------------------------
inline double Vector2D_Magnitude(Vector2D *u)
{
    return(sqrt(u->x * u->x + u->y * u->y ));
}


//--- Calculate distance between two 3D points ------------------------------
inline double Vector3D_Distance(Vector3D *u, Vector3D *v)
{
    double tmp = (u->x-v->x)*(u->x-v->x) + (u->y-v->y)*(u->y-v->y) +
        (u->z-v->z)*(u->z-v->z);
    return(sqrt(tmp));
}

//--- Calculate distance between two 3D points ------------------------------
inline double Vector2D_Distance(Vector2D *u, Vector2D *v)
{
    double tmp = (u->x-v->x)*(u->x-v->x) + (u->y-v->y)*(u->y-v->y);
    return(sqrt(tmp));
}

inline void Negate_Vector3D(vector3D *a)
{
    a->x = - a->x;
    a->y = - a->y;
    a->z = - a->z;
}

inline void Negate_Vector2D(vector3D *a)
{
    a->x = - a->x;
    a->y = - a->y;
}

//--- Calculate cross-product -----------------------------------------------
inline void Cross_Product(vector3D *u, vector3D *v, vector3D *cp)
{
    cp->x = (u->y * v->z) - (u->z * v->y);
    cp->y = (u->z * v->x) - (u->x * v->z);
    cp->z = (u->x * v->y) - (u->y * v->x);
}


//--- Calculate dot-product -------------------------------------------------
inline double Dot_Product3D(vector3D *u, vector3D *v)
{
    return((u->x * v->x) + (u->y * v->y) + (u->z * v->z));
}

inline double Dot_Product2D(vector2D *u, vector2D *v)
{
    return((u->x * v->x) + (u->y * v->y));
}

void    Make_Plane3D(vector3D *a, vector3D *b, vector3D *c, Plane* result);
void    Normalize_Vector3D(vector3D *u);
void    Normalize_Vector2D(vector2D *u);



void Plane_Face_Point(Plane* plane, Vertex3D* point);
void Plane_NotFace_Point(Plane* plane, Vertex3D* point);
*/


#endif

