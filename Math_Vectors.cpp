//#include "stdafx.h"
//****************************************************************************
//
// File         : vector.cpp
//
// Description  : Misc 2D/3D vector functions.
//
//
//****************************************************************************


#include <math.h>
#include "Math_Vectors.h"

/*

//--- Normalize 3d vector ---------------------------------------------------
void Normalize_Vector3D(Vector3D *u)
{

    double length = Vector3D_Magnitude(u);
    if (length != 0) {
        u->x /= length;
        u->y /= length;
        u->z /= length;
    }
    else
    {
        u->x = u->y = u->z = 0;
    }    
}

//--- Normalize 2d vector ---------------------------------------------------
void Normalize_Vector2D(Vector2D *u)
{

    double length = Vector2D_Magnitude(u);
    if (length != 0) {
        u->x /= length;
        u->y /= length;
    }
    else
    {
        u->x = u->y = 0;
    }    
}



//--- Determine at which side of a line the point x,y lies ------------------

int WhichSide_line2D(Line2D *v, double x, double y)
{
    double s = (v->y1 - y)*(v->x2 - v->x1) - (v->x1 - x)*(v->y2 - v->y1);
    if (s < 0) return -1;
    else if (s > 0) return 1;
    else return 0;
}

//--- Calculate line intersection -------------------------------------------

#define VERY_SMALL 0.00001

bool Seg_Line_Intersect(Line2D *s, Line2D *l, Vertex2D *isect)
{
    double   denom;
    double   r;

    denom = (s->x2 - s->x1) * (l->y2 - l->y1) -
        (s->y2 - s->y1) * (l->x2 - l->x1);
    if (fabs(denom) < VERY_SMALL) return(false);

    r = ((s->y1 - l->y1) * (l->x2 - l->x1) -
        (s->x1 - l->x1) * (l->y2 - l->y1)) / denom;

    if ((r < -VERY_SMALL) || (r > 1+VERY_SMALL)) return(false);

    isect->x = s->x1 + r * (s->x2 - s->x1);
    isect->y = s->y1 + r * (s->y2 - s->y1);

    return(true);
}



//--- ������ ������� �������� �� 3 �����
void Make_Plane3D(Vertex3D *a, Vertex3D *b, Vertex3D *c, Plane* result)
{
    Vector3D v1,v2;

    Make_Vector3D(a, b, &v1); // v1 = "b -> a"
    Make_Vector3D(a, c, &v2); // v2 = "c -> a"

    Cross_Product(&v1, &v2, &(result->normal)); // ������� ��������� ������

    Normalize_Vector3D(&(result->normal));

    result->d = - Dot_Product3D(&(result->normal), a);
};




// ���� ��������� �� ���� � ���� ��� ������ �����
void Plane_Face_Point(Plane* plane, Vertex3D* point)
{
     double dist = (plane->normal.x * point->x) +
                   (plane->normal.y * point->y) +
                   (plane->normal.z * point->z) + plane->d;

     // ��� �� � �������� ��� �������
     if (dist<0)
     {
        // ��������
        Negate_Vector3D(&(plane->normal));
        plane->d = - plane->d;
     }
}


// ���� ��������� �� �� ���� � ���� ��� ������ �����
void Plane_NotFace_Point(Plane* plane, Vertex3D* point)
{
      double dist = (plane->normal.x * point->x) +
                    (plane->normal.y * point->y) +
                    (plane->normal.z * point->z) + plane->d;

     // ��� � �������� ��� �������
     if (dist>0)
     {
        // ��������
        Negate_Vector3D(&(plane->normal));
        plane->d = - plane->d;
     }
}

*/
