//****************************************************************************
//
// File         : Matrix.h
//
// Description  : 4x4 Matrix math header file.
//
//
//****************************************************************************

#ifndef MATRIX_H
#define MATRIX_H

#include "Math_Vectors.h"


typedef double Matrix[4][4];

// Set a given matrix to the null matrix
extern void Set_Null(Matrix mat);

// Set a given matrix to the identity matrix
// (Space Reset)
extern void Set_Identity(Matrix mat);

// Multiply two matrices and store result in a third one
extern void Matrix_Mult(Matrix a, Matrix b, Matrix res);

// Multiplication of a matrix with a rotation matrix
// (Space Rotation)
extern void Rotate_Matrix_X(Matrix mat, double a);
extern void Rotate_Matrix_Y(Matrix mat, double a);
extern void Rotate_Matrix_Z(Matrix mat, double a);

// Multiplication of a matrix with a translation matrix
// (Space Translation)
extern void Translate_Matrix(Matrix mat, double x, double y, double z);


// This transformation:
// The Matrix's space coordinates -> to World Space coordinates
inline void Transform_Vector_From_Matrix_2D(Matrix matrix, vector2D& vec, vector2D& dest)
{
    dest.x = vec.x * matrix[0][0] + vec.y * matrix[1][0] + //vec->z * matrix[2][0] +
             matrix[3][0];

    dest.y = vec.x * matrix[0][1] + vec.y * matrix[1][1] + //vec->z * matrix[2][1] +
             matrix[3][1];

    /*    dest.z = vec->x * matrix[0][2] + vec->y * matrix[1][2] + vec->z * matrix[2][2] +
                matrix[3][2];  */
}

inline void Transform_Vector_From_Matrix_3D(Matrix matrix, vector3D& vec, vector3D& dest)
{
    dest.x = vec.x * matrix[0][0] + vec.y * matrix[1][0] + vec.z * matrix[2][0] +
             matrix[3][0];

    dest.y = vec.x * matrix[0][1] + vec.y * matrix[1][1] + vec.z * matrix[2][1] +
             matrix[3][1];

    dest.z = vec.x * matrix[0][2] + vec.y * matrix[1][2] + vec.z * matrix[2][2] +
             matrix[3][2];
}



// World Space -> Matrix space
inline void Transform_Vector_To_Matrix_2D(Matrix matrix, vector2D& src, vector2D& dest)
{
    vector2D tmp;

    tmp.x = src.x - matrix[3][0];
    tmp.y = src.y - matrix[3][1];
//   tmp.z = src->z - matrix[3][2];

    dest.x = tmp.x*matrix[0][0] + tmp.y*matrix[0][1];// + tmp.z*matrix[0][2];
    dest.y = tmp.x*matrix[1][0] + tmp.y*matrix[1][1];// + tmp.z*matrix[1][2];
//   dest->z = tmp.x*matrix[2][0] + tmp.y*matrix[2][1] + tmp.z*matrix[2][2];
}


inline void Transform_Vector_To_Matrix_3D(Matrix matrix, vector3D& src, vector3D& dest)
{
    vector3D tmp;

    tmp.x   = src.x - matrix[3][0];
    tmp.y   = src.y - matrix[3][1];
    tmp.z   = src.z - matrix[3][2];

    dest.x  = tmp.x*matrix[0][0] + tmp.y*matrix[0][1] + tmp.z*matrix[0][2];
    dest.y  = tmp.x*matrix[1][0] + tmp.y*matrix[1][1] + tmp.z*matrix[1][2];
    dest.z  = tmp.x*matrix[2][0] + tmp.y*matrix[2][1] + tmp.z*matrix[2][2];
}



#endif

