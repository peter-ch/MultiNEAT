//#include "stdafx.h"
//****************************************************************************
//
// File         : Matrix.cpp
//
// Description  : 4x4 Matrix math routines.
//
//
//****************************************************************************


#include <math.h>
#include "Math_Matrix.h"

//////////////////////////////////////
// Matrix is a 4x4 array of floats  //
// [row][column] = pos              //
//////////////////////////////////////


// Set the given matrix to the null matrix
void Set_Null(Matrix mat)
{
    for(register char i=0; i<4; i++)
        for(register char j=0; j<4; j++)
            mat[static_cast<int>(i)][static_cast<int>(j)]=0;
}


// Set the given matrix to the identity matrix
void Set_Identity(Matrix mat)
{
    for(register char i=0; i<4; i++)
        for(register char j=0; j<4; j++)
            mat[static_cast<int>(i)][static_cast<int>(j)]=0;

    mat[0][0] = 1;
    mat[1][1] = 1;
    mat[2][2] = 1;
    mat[3][3] = 1;
}


// Create a translation matrix
inline void Set_Translation(Matrix mat, double x, double y, double z)
{
    Set_Identity(mat);

    mat[3][0] = x;
    mat[3][1] = y;
    mat[3][2] = z;
}


// Create a rotation matrix for rotation along the X axis
inline void Set_Rotation_X(Matrix mat, double a)
{
    Set_Identity(mat);

    mat[1][1] = cos(a);
    mat[1][2] = sin(a);
    mat[2][1] = -sin(a);
    mat[2][2] = cos(a);
}

// Create a rotation matrix for rotation along the Y axis
inline void Set_Rotation_Y(Matrix mat, double a)
{
    Set_Identity(mat);

    mat[0][0] = cos(a);
    mat[0][2] = -sin(a);
    mat[2][0] = sin(a);
    mat[2][2] = cos(a);
}

// Create a rotation matrix for rotation along the Z axis
inline void Set_Rotation_Z(Matrix mat, double a)
{
    Set_Identity(mat);

    mat[0][0] = cos(a);
    mat[0][1] = sin(a);
    mat[1][0] = -sin(a);
    mat[1][1] = cos(a);
}


// Multiply two matrices and store the result in a third one
void Matrix_Mult(Matrix a, Matrix b, Matrix res)
{
    register unsigned char col,row;

    for(row=0; row<4; row++)    // for each row in source matrix
        for(col=0; col<4; col++) // for each column dest matrix
        {

            res[row][col] = a[row][0]*b[0][col] + a[row][1]*b[1][col] + a[row][2]*b[2][col] + a[row][3]*b[3][col];
        }
}


// Rotate a matrix using above functions
void Rotate_Matrix_X(Matrix mat, double a)
{
    Matrix temp,res;

    Set_Rotation_X(temp,a);

    Matrix_Mult(temp,mat,res);

    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            mat[i][j] = res[i][j];

}

void Rotate_Matrix_Y(Matrix mat, double a)
{
    Matrix temp,res;

    Set_Rotation_Y(temp,a);

    Matrix_Mult(temp,mat,res);

    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            mat[i][j] = res[i][j];
}

void Rotate_Matrix_Z(Matrix mat, double a)
{
    Matrix temp,res;

    Set_Rotation_Z(temp,a);

    Matrix_Mult(temp,mat,res);
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            mat[i][j] = res[i][j];
}

// Translate a matrix using above functions
void Translate_Matrix(Matrix mat, double x, double y, double z)
{
    Matrix temp,res;

    Set_Translation(temp,x,y,z);

    Matrix_Mult(temp,mat,res);

    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            mat[i][j] = res[i][j];
}



