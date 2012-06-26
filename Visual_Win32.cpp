#if 0
/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2007, NEAT Sciences Ltd.
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Visual_Win32.cpp
// Description: Genome visualization routines for Microsoft Windows
///////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "Visual_Win32.h"
#include "Math_Matrix.h"

using namespace NS;
using namespace NEAT;


// Good to find out the direction of the connections
#define ARROW_ANGLE 0.025
//#define DISPLAY_INNOV_NUMBERS


////////////////////////////////////////////////
// Forward Vector & Matrix stuff declarations //
////////////////////////////////////////////////

#include <math.h>


#define VERY_SMALL 0.000001



#define sqr(x) ((x)*(x))
/*
class vector2D
{
public:
	double  x, y;
	double magnitude() { return sqrt( sqr(x) + sqr(y) ); }
	double sqr_magnitude() { return ( sqr(x) + sqr(y) ); }
	void normalize()  { double mag = magnitude(); if (mag != 0) { x /= mag; y /= mag; } }
	// Initialize
	vector2D() { x = 0; y = 0; }
	vector2D(double xx, double yy) { x = xx; y = yy; }
	// Make it point from p1 to p2
	vector2D(vector2D& p1, vector2D& p2) { 	x = p2.x - p1.x; y = p2.y - p1.y; }
	vector2D plus(vector2D& vec)  { vector2D tmp; tmp.x = x + vec.x; tmp.y = y + vec.y; return tmp; }
	vector2D minus(vector2D& vec) { vector2D tmp; tmp.x = x - vec.x; tmp.y = y - vec.y; return tmp; }
	void add(vector2D& vec) { x += vec.x; y += vec.y; }
	void sub(vector2D& vec) { x -= vec.x; y -= vec.y; }
	double distance_to(vector2D& vec) { return sqrt( sqr(vec.x - x) + sqr(vec.y - y) ); }
	double sqr_distance_to(vector2D& vec) { return ( sqr(vec.x - x) + sqr(vec.y - y) ); }
	double dot(vector2D& vec) { return ((x * vec.x) + (y * vec.y)); }
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
	void negate() {	x = -x;	y = -y; }
};

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))

class line2D
{
public:
	vector2D p1, p2;
//  Initialize from two points
	line2D(vector2D& i1, vector2D& i2) { p1 = i1; p2 = i2; }
	line2D() { p1.x = 0; p1.y = 0; p2.x = 0; p2.y = 0; }
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
			isect = p1;
			isect.x += t*D.x;
			isect.y += t*D.y;
			if (isect.sqr_distance_to(p1) > p2.sqr_distance_to(p1)) return false;
			else return true;
		}
		return true;
	}
};

*/
/*
typedef double Matrix[4][4];

// Set a given matrix to the null matrix
void Set_Null(Matrix mat);

// Set a given matrix to the identity matrix
// (Space Reset)
void Set_Identity(Matrix mat);

// Multiply two matrices and store result in a third one
void Matrix_Mult(Matrix a, Matrix b, Matrix res);

// Multiplication of a matrix with a rotation matrix
// (Space Rotation)
void Rotate_Matrix_X(Matrix mat, double a);
void Rotate_Matrix_Y(Matrix mat, double a);
void Rotate_Matrix_Z(Matrix mat, double a);

// Multiplication of a matrix with a translation matrix
// (Space Translation)
void Translate_Matrix(Matrix mat, double x, double y, double z);

// This transformation:
// The Matrix's space coordinates -> to World Space coordinates
inline void Transform_Vector_From_Matrix_2D(Matrix matrix, vector2D& vec, vector2D& dest)
{
    dest.x = vec.x * matrix[0][0] + vec.y * matrix[1][0] + matrix[3][0];

    dest.y = vec.x * matrix[0][1] + vec.y * matrix[1][1] + matrix[3][1];
}

// World Space -> Matrix space
inline void Transform_Vector_To_Matrix_2D(Matrix matrix, vector2D& src, vector2D& dest)
{
   vector2D tmp;

   tmp.x = src.x - matrix[3][0];
   tmp.y = src.y - matrix[3][1];

   dest.x = tmp.x*matrix[0][0] + tmp.y*matrix[0][1];
   dest.y = tmp.x*matrix[1][0] + tmp.y*matrix[1][1];
}
*/







///////////////////////////
// Genome Drawing code   //
///////////////////////////

// Maximum possible depth of genome
#define MAX_DEPTH 64
#define XPOS_ALTER 0
#define YPOS_ALTER 0

void Draw_Genome(NS::NEAT::Genome& genome, HDC& dc, int xpos, int ypos, int rect_x_size, int rect_y_size, DrawingType drawing_type, int neuron_radius, int max_line_thickness, int arrow_spike_length)
{
    //////////////////////////////////
    // Prepare to draw Genome
    // initialize neuron positions


    unsigned int i;
    int xxsize = rect_x_size;
    int xxpos;
    double depth;
    double depth_inc = 1.0 / MAX_DEPTH;

    if ((drawing_type == STANDART) || (drawing_type == CPPN))
    {
        // Normal drawing

        // process X coords
        // for every possible depth
        for(depth=0; depth<=1.0; depth+=depth_inc)
        {
            // count how many nodes at this depth
            int neuron_count=0;
            for(i=0; i<genome.NumNeurons(); i++)
                if (genome.GetNeuronByIndex(i).SplitY() == depth) neuron_count++;

            // skip this depth if there are no nodes
            if (neuron_count == 0) continue;

            // calculate X positions for nodes at this depth
            int j=0;
            xxpos = (rect_x_size / (1 + neuron_count));
            for(i=0; i<genome.NumNeurons(); i++)
            {
                if (genome.GetNeuronByIndex(i).SplitY() == depth)
                {
                    genome.SetNeuronX(i, xpos + xxpos + j*(rect_x_size/(2 + neuron_count)));
                    j++;
                }
            }
        }

        // process Y coords
        for(i=0; i<genome.NumNeurons(); i++)
        {
            genome.SetNeuronY(i, ypos + static_cast<int>(genome.GetNeuronByIndex(i).SplitY() * (rect_y_size-neuron_radius) + neuron_radius));

            // alter position
            /*if (genome.m_NeuronGenes[i].m_Type == NS::NEAT::HIDDEN)
            {
            	genome.m_NeuronGenes[i].x += static_cast<int>((RandFloat() * XPOS_ALTER) - XPOS_ALTER/2);
            	genome.m_NeuronGenes[i].y += static_cast<int>((RandFloat() * YPOS_ALTER) - YPOS_ALTER/2);
            }

            // move NS::NEAT::OUTPUTs up a little bit
            if (genome.m_NeuronGenes[i].m_SplitY == 1.0)
            {
            	genome.m_NeuronGenes[i].y -= neuron_radius * 2.0;
            }*/
        }
    }

    if (drawing_type == CTRNN)
    {
        //CTRNN drawing

        double x=rect_x_size/2, y=rect_y_size/2, x_radius=rect_x_size/2 - 50, y_radius=rect_y_size/2 - 50;
        double ang;
        int cur_node = 0;
        for(ang = 0.0; ang < 2*3.14+1; ang += 2*3.14/genome.NumNeurons(), cur_node++)
        {
            if (cur_node == genome.NumNeurons())
                break;

            genome.SetNeuronXY(cur_node, xpos + static_cast<int>(x - y_radius * sin(ang)), ypos + static_cast<int>(y - y_radius * cos(ang)));
        }
    }


    ///////////////////////////////////////
    // Neuron X,Y positions are now set
    // Let's draw!
    ///////////////////////////////////////

    // Determine max weight
    double MAX_WEIGHT_MAGNITUDE = DBL_MIN;
    for(i=0; i<genome.NumLinks(); i++)
    {
        const double t_Weight = abs(genome.GetLinkByIndex(i).GetWeight());
        // visit each node and check the weight of incoming links
        if (t_Weight > MAX_WEIGHT_MAGNITUDE) MAX_WEIGHT_MAGNITUDE = t_Weight;
    }


    // First draw all connections
    for(i=0; i<genome.NumLinks(); i++)
    {
        HPEN pen;
        const NS::NEAT::LinkGene t_LinkGene = genome.GetLinkByIndex(i);
        const double t_LinkWeight = t_LinkGene.GetWeight();
        double thickness = abs(t_LinkWeight);
        double mag = abs(t_LinkWeight);

        if (MAX_WEIGHT_MAGNITUDE != 0)
        {
            Scale(thickness, 0, MAX_WEIGHT_MAGNITUDE, 1, max_line_thickness);
            Scale(mag, 0, MAX_WEIGHT_MAGNITUDE, 100, 255);
        }
        else
        {
            thickness = max_line_thickness;
            mag = 255;
        }

        //thickness *= 1.0;
        //mag = 255;

        Clamp(thickness, 1, max_line_thickness);
        //Clamp(mag, 0.0, 1.0);

        if (!(t_LinkGene.IsRecurrent()))
        {
            if (t_LinkWeight > 0)
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(static_cast<int>(mag), 0, 0)); // RED POSITIVE
            else
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(0, 0, static_cast<int>(mag))); // BLUE NEGATIVE
        }
        else
        {
            // RECURRENT
            if (t_LinkWeight > 0)
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(static_cast<int>(mag), static_cast<int>(mag), static_cast<int>(mag))); // WHITE POSITIVE
            else
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(0, static_cast<int>(mag), 0)); // GREEN NEGATIVE
        }

        double max_rad = neuron_radius;

        SelectObject(dc, pen);

        // If this is a looped recurrent link, draw an ellipse to represent it
        if ( t_LinkGene.FromNeuronID() == t_LinkGene.ToNeuronID() )
        {
            // this is the background color! (of the recurrent self-loop)
            HBRUSH brr = CreateSolidBrush(RGB(0, 0, 0));
            SelectObject(dc, brr);

            int idx = genome.GetNeuronIndex(genome.GetLinkByIndex(i).FromNeuronID());

            //Ellipse(dc, genome.GetNeuronByIndex(idx).x - max_rad-1, genome.GetNeuronByIndex(idx).y,
            //genome.GetNeuronByIndex(idx).x + max_rad, genome.GetNeuronByIndex(idx).y + max_rad*2);

            // draw a polygon
            int base_x = genome.GetNeuronByIndex(idx).x, base_y = genome.GetNeuronByIndex(idx).y;
            MoveToEx(dc, base_x, base_y, NULL);
            LineTo(dc, base_x - neuron_radius*3, base_y + neuron_radius*3);
            LineTo(dc, base_x + neuron_radius*3, base_y + static_cast<int>(neuron_radius*3.4));
            LineTo(dc, base_x, base_y);


            DeleteObject(brr);
        }



        // Draw the arrows of the lines
        // link->out node is the place with the arrows
        vector2D A, B, C;
        vector2D isect;
        int aa,bb;
        line2D ol;

        vector2D result;
        Matrix matrix;

        vector2D tmp,t;

        // Initialize points
        LinkGene l = t_LinkGene;
        NeuronGene n = genome.GetNeuronByID(l.FromNeuronID());

        A.x = n.x;
        A.y = n.y;

        n = genome.GetNeuronByID(l.ToNeuronID());
        B.x = n.x;
        B.y = n.y;

        C.x = B.x;
        C.y = B.y;

        // draw the arrow .. spikes ;)
        for(double ang=-ARROW_ANGLE; ang <= ARROW_ANGLE; ang += ARROW_ANGLE * 2)
        {
            B.x = C.x;
            B.y = C.y;

            // init matrix
            Set_Identity(matrix);
            // translate matrix to the vector's position (origin)
            matrix[3][0] = B.x;
            matrix[3][1] = B.y;
            matrix[3][2] = 0;
            // Move B to the edge of the circle
            Transform_Vector_To_Matrix_2D(matrix, A, t);
            t.normalize();
            B.x = C.x + t.x*max_rad;
            B.y = C.y + t.y*max_rad;


            // init matrix
            Set_Identity(matrix);
            // rotate the coordinate system
            Rotate_Matrix_Z(matrix, ang);
            // translate matrix to the vector's position (origin)
            matrix[3][0] = B.x;
            matrix[3][1] = B.y;
            matrix[3][2] = 0;
            // Scale matrix
            matrix[0][0] *= 0.11;
            matrix[1][1] *= 0.11;

            // transform A into the translated coordinate system
            Transform_Vector_To_Matrix_2D(matrix, A, result);
            result.normalize();
            // Draw the line (transformed back)
            MoveToEx(dc, static_cast<int>(B.x), static_cast<int>(B.y), NULL);
            aa = static_cast<int>(B.x + (result.x * arrow_spike_length));
            bb = static_cast<int>(B.y + (result.y * arrow_spike_length));
            LineTo(dc, aa, bb);
        }

        n = genome.GetNeuronByID(l.FromNeuronID());

        // Draw line of connection
        MoveToEx(dc, static_cast<int>(C.x - tmp.x*max_rad), static_cast<int>(C.y - tmp.y*max_rad), NULL);
        LineTo(dc, n.x, n.y);

        // I want to see the innovation number of the line
#ifdef DISPLAY_INNOV_NUMBERS
        char str[8];
        sprintf(str, "%d", t_LinkGene.InnovationID());
        TextOutA(dc, static_cast<int>((C.x - tmp.x*max_rad) + n.x)/2, static_cast<int>((C.y - tmp.y*max_rad) + n.y)/2, str, strlen(str));
#endif
        DeleteObject(pen);
    } // end for all links


    // Now draw all nodes
    double radius = neuron_radius;

    for(i=0; i<genome.NumNeurons(); i++)
    {
        HBRUSH br1 = CreateSolidBrush(RGB(0, 0, 0));
        HPEN pen1;
        const NS::NEAT::NeuronGene t_Neuron = genome.GetNeuronByIndex(i);

        if (t_Neuron.Type() == NS::NEAT::INPUT)
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 0, 0));  // contour
        else if (t_Neuron.Type() == NS::NEAT::HIDDEN)
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 255, 0));  // contour
        else if (t_Neuron.Type() == NS::NEAT::OUTPUT)
            pen1 = CreatePen(PS_SOLID, 1, RGB(0, 255, 0));  // contour
        else
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));  // contour


        // Draw contour
        SelectObject(dc, pen1);
        SelectObject(dc, br1);
        Ellipse(dc, t_Neuron.x - static_cast<int>(radius-1), t_Neuron.y - static_cast<int>(radius-1),
                t_Neuron.x + static_cast<int>(radius), t_Neuron.y + static_cast<int>(radius));
        DeleteObject(pen1);
        DeleteObject(br1);

        // Display the node's number under the node
        char str[64];
        char type[64];
        char stype[64];
        stype[0]=0;
        type[0]=0;

        if ((t_Neuron.Type() != NS::NEAT::INPUT) && (t_Neuron.Type() != BIAS))
            switch(t_Neuron.m_ActFunction)
            {
            case SIGNED_SIGMOID:
                sprintf(type, "S +/-");
                break;

            case UNSIGNED_SIGMOID:
                sprintf(type, "S");
                break;

            case TANH:
                sprintf(type, "T");
                break;

            case TANH_CUBIC:
                sprintf(type, "T3");
                break;

            case SIGNED_STEP:
                sprintf(type, "St +/-");
                break;

            case UNSIGNED_STEP:
                sprintf(type, "St");
                break;

            case SIGNED_GAUSS:
                sprintf(type, "G +/-");
                break;

            case UNSIGNED_GAUSS:
                sprintf(type, "G");
                break;

            case ABS:
                sprintf(type, "A");
                break;

            case SIGNED_SINE:
                sprintf(type, "Si +/-");
                break;

            case UNSIGNED_SINE:
                sprintf(type, "Si");
                break;

            case SIGNED_SQUARE:
                sprintf(type, "Sq +/-");
                break;

            case UNSIGNED_SQUARE:
                sprintf(type, "Sq");
                break;

            case LINEAR:
                sprintf(type, "L");
                break;

            default:
                sprintf(type, "S");
                break;
            }

        sprintf(str, "%s\n",
                //static_cast<int>(t_Neuron.ID()),
                type//,
                //t_Neuron.m_A
               );

        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 0));
        int len = static_cast<int>(strlen(str)-1);
        len *= 6; // pixels for a char
        len /= 2;

        if (drawing_type == CPPN)
            TextOut(dc, t_Neuron.x-len, t_Neuron.y + static_cast<int>(radius) + 5, str, static_cast<int>(strlen(str)-1));
    }
}































void Draw_NN(NS::NEAT::NeuralNetwork& net, HDC& dc, int xpos, int ypos, int rect_x_size, int rect_y_size, DrawingType drawing_type, int neuron_radius, int max_line_thickness, int arrow_spike_length)
{
    //////////////////////////////////
    // Prepare to draw Genome
    // initialize neuron positions
    for(int i=0; i<net.m_neurons.size(); i++)
    {
        net.m_neurons[i].m_x;
        net.m_neurons[i].m_y;
        net.m_neurons[i].m_z;
    }


    unsigned int i;
    int xxsize = rect_x_size;
    int xxpos;
    double depth;
    double depth_inc = 1.0 / MAX_DEPTH;

    if ((drawing_type == STANDART) || (drawing_type == CPPN))
    {
        // Normal drawing

        // process X coords
        // for every possible depth
        for(depth=0; depth<=1.0; depth+=depth_inc)
        {
            // count how many nodes at this depth
            int neuron_count=0;
            for(i=0; i<net.m_neurons.size(); i++)
                if (net.m_neurons[i].m_split_y == depth) neuron_count++;

            // skip this depth if there are no nodes
            if (neuron_count == 0) continue;

            // calculate X positions for nodes at this depth
            int j=0;
            xxpos = (rect_x_size / (1 + neuron_count));
            for(i=0; i<net.m_neurons.size(); i++)
            {
                if (net.m_neurons[i].m_split_y == depth)
                {
                    //net.SetNeuronX(i, xpos + xxpos + j*(rect_x_size/(2 + neuron_count)));
                    net.m_neurons[i].m_x = xpos + xxpos + j*(rect_x_size/(2 + neuron_count));
                    j++;
                }
            }
        }

        // process Y coords
        for(i=0; i<net.m_neurons.size(); i++)
        {
            //net.SetNeuronY(i, ypos + static_cast<int>(net.m_neurons[i].m_split_y * (rect_y_size-neuron_radius) + neuron_radius));
            net.m_neurons[i].m_y = ypos + static_cast<int>(net.m_neurons[i].m_split_y * (rect_y_size-neuron_radius) + neuron_radius);

            // alter position
            /*if (net.m_NeuronGenes[i].m_Type == NS::NEAT::HIDDEN)
            {
            	net.m_NeuronGenes[i].x += static_cast<int>((RandFloat() * XPOS_ALTER) - XPOS_ALTER/2);
            	net.m_NeuronGenes[i].y += static_cast<int>((RandFloat() * YPOS_ALTER) - YPOS_ALTER/2);
            }

            // move NS::NEAT::OUTPUTs up a little bit
            if (net.m_NeuronGenes[i].m_SplitY == 1.0)
            {
            	net.m_NeuronGenes[i].y -= neuron_radius * 2.0;
            }*/
        }
    }

    if (drawing_type == CTRNN)
    {
        //CTRNN drawing

        double x=rect_x_size/2, y=rect_y_size/2, x_radius=rect_x_size/2 - 50, y_radius=rect_y_size/2 - 50;
        double ang;
        int cur_node = 0;
        for(ang = 0.0; ang < 2*3.14+1; ang += 2*3.14/net.m_neurons.size(), cur_node++)
        {
            if (cur_node == net.m_neurons.size())
                break;

            //net.SetNeuronXY(cur_node, xpos + static_cast<int>(x - y_radius * sin(ang)), ypos + static_cast<int>(y - y_radius * cos(ang)));
            net.m_neurons[cur_node].m_x = xpos + static_cast<int>(x - y_radius * sin(ang));
            net.m_neurons[cur_node].m_y = ypos + static_cast<int>(y - y_radius * cos(ang));
        }
    }


    if (drawing_type == SUBSTRATE)
    {
        // HYPERNEAT Drawing

        for(i=0; i<net.m_neurons.size(); i++)
        {
            vector3D tmp,a;

            a.x =  net.m_neurons[i].m_sx;
            a.y =  net.m_neurons[i].m_sy;

            net.m_neurons[i].m_x = xpos + (( a.x ) * (rect_y_size/2.15))/1 + rect_x_size/2;
            net.m_neurons[i].m_y = ypos + (( a.y ) * (rect_y_size/2.15))/1 + rect_y_size/2;
            //net->all_nodes[i]->rad  = (( 2 * a.x ) * (FIELD_Y/2.15))/1 + FIELD_X/2;

            // Tinker with the node pos
            //net->all_nodes[i]->xpos += (RandFloat() * XPOS_ALTER) - XPOS_ALTER/2;
            //net->all_nodes[i]->ypos += (RandFloat() * YPOS_ALTER) - YPOS_ALTER/2;
        }
    }

    ///////////////////////////////////////
    // Neuron X,Y positions are now set
    // Let's draw!
    ///////////////////////////////////////

    // Determine max weight
    double MAX_WEIGHT_MAGNITUDE = DBL_MIN;
    for(i=0; i<net.m_connections.size(); i++)
    {
        const double t_Weight = abs(net.m_connections[i].m_weight);
        // visit each node and check the weight of incoming links
        if (t_Weight > MAX_WEIGHT_MAGNITUDE) MAX_WEIGHT_MAGNITUDE = t_Weight;
    }


    // First draw all connections
    for(i=0; i<net.m_connections.size(); i++)
    {
        HPEN pen;
        const NS::NEAT::Connection t_LinkGene = net.m_connections[i];
        const double t_LinkWeight = t_LinkGene.m_weight;
        double thickness = abs(t_LinkWeight);
        double mag = abs(t_LinkWeight);

        if (MAX_WEIGHT_MAGNITUDE != 0)
        {
            Scale(thickness, 0, MAX_WEIGHT_MAGNITUDE, 1, max_line_thickness);
            Scale(mag, 0, MAX_WEIGHT_MAGNITUDE, 100, 255);
        }
        else
        {
            thickness = max_line_thickness;
            mag = 255;
        }

        //thickness *= 1.0;
        //mag = 255;

        Clamp(thickness, 1, max_line_thickness);
        //Clamp(mag, 0.0, 1.0);

        if (!(t_LinkGene.m_recur_flag))
        {
            if (t_LinkWeight > 0)
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(static_cast<int>(mag), 0, 0)); // RED POSITIVE
            else
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(0, 0, static_cast<int>(mag))); // BLUE NEGATIVE
        }
        else
        {
            // RECURRENT
            if (t_LinkWeight > 0)
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(static_cast<int>(mag), static_cast<int>(mag), static_cast<int>(mag))); // WHITE POSITIVE
            else
                pen = CreatePen(PS_SOLID, static_cast<int>(thickness), RGB(0, static_cast<int>(mag), 0)); // GREEN NEGATIVE
        }

        double max_rad = neuron_radius;

        SelectObject(dc, pen);

        // If this is a looped recurrent link, draw an ellipse to represent it
        if ( t_LinkGene.m_source_neuron_idx == t_LinkGene.m_target_neuron_idx )
        {
            // this is the background color! (of the recurrent self-loop)
            HBRUSH brr = CreateSolidBrush(RGB(0, 0, 0));
            SelectObject(dc, brr);

            int idx = t_LinkGene.m_source_neuron_idx;

            //Ellipse(dc, net.GetNeuronByIndex(idx).x - max_rad-1, net.GetNeuronByIndex(idx).y,
            //net.GetNeuronByIndex(idx).x + max_rad, net.GetNeuronByIndex(idx).y + max_rad*2);

            // draw a polygon
            int base_x = net.m_neurons[idx].m_x, base_y = net.m_neurons[idx].m_y;
            MoveToEx(dc, base_x, base_y, NULL);
            LineTo(dc, base_x - neuron_radius*3, base_y + neuron_radius*3);
            LineTo(dc, base_x + neuron_radius*3, base_y + static_cast<int>(neuron_radius*3.4));
            LineTo(dc, base_x, base_y);


            DeleteObject(brr);
        }



        // Draw the arrows of the lines
        // link->out node is the place with the arrows
        vector2D A, B, C;
        vector2D isect;
        int aa,bb;
        line2D ol;

        vector2D result;
        Matrix matrix;

        vector2D tmp,t;

        // Initialize points
        Connection l = t_LinkGene;
        Neuron n = net.m_neurons[l.m_source_neuron_idx];

        A.x = n.m_x;
        A.y = n.m_y;

        n = net.m_neurons[l.m_target_neuron_idx];
        B.x = n.m_x;
        B.y = n.m_y;

        C.x = B.x;
        C.y = B.y;

        // draw the arrow .. spikes ;)
        for(double ang=-ARROW_ANGLE; ang <= ARROW_ANGLE; ang += ARROW_ANGLE * 2)
        {
            B.x = C.x;
            B.y = C.y;

            // init matrix
            Set_Identity(matrix);
            // translate matrix to the vector's position (origin)
            matrix[3][0] = B.x;
            matrix[3][1] = B.y;
            matrix[3][2] = 0;
            // Move B to the edge of the circle
            Transform_Vector_To_Matrix_2D(matrix, A, t);
            t.normalize();
            B.x = C.x + t.x*max_rad;
            B.y = C.y + t.y*max_rad;


            // init matrix
            Set_Identity(matrix);
            // rotate the coordinate system
            Rotate_Matrix_Z(matrix, ang);
            // translate matrix to the vector's position (origin)
            matrix[3][0] = B.x;
            matrix[3][1] = B.y;
            matrix[3][2] = 0;
            // Scale matrix
            matrix[0][0] *= 0.11;
            matrix[1][1] *= 0.11;

            // transform A into the translated coordinate system
            Transform_Vector_To_Matrix_2D(matrix, A, result);
            result.normalize();
            // Draw the line (transformed back)
            MoveToEx(dc, static_cast<int>(B.x), static_cast<int>(B.y), NULL);
            aa = static_cast<int>(B.x + (result.x * arrow_spike_length));
            bb = static_cast<int>(B.y + (result.y * arrow_spike_length));
            LineTo(dc, aa, bb);
        }

        n = net.m_neurons[l.m_source_neuron_idx];

        // Draw line of connection
        MoveToEx(dc, static_cast<int>(C.x - tmp.x*max_rad), static_cast<int>(C.y - tmp.y*max_rad), NULL);
        LineTo(dc, n.m_x, n.m_y);

        // I want to see the innovation number of the line
        DeleteObject(pen);
    } // end for all links


    // Now draw all nodes
    double radius = neuron_radius;

    for(i=0; i<net.m_neurons.size(); i++)
    {
        HBRUSH br1 = CreateSolidBrush(RGB(0, 0, 0));
        HBRUSH br2;

        HPEN pen1;
        HPEN pen2 = CreatePen(PS_SOLID, 1, RGB(0, 0, 0));
        const NS::NEAT::Neuron t_Neuron = net.m_neurons[i];

        if (t_Neuron.m_type == NS::NEAT::INPUT)
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 0, 0));  // contour
        else if (t_Neuron.m_type == NS::NEAT::HIDDEN)
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 255, 0));  // contour
        else if (t_Neuron.m_type == NS::NEAT::OUTPUT)
            pen1 = CreatePen(PS_SOLID, 1, RGB(0, 255, 0));  // contour
        else
            pen1 = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));  // contour


        // Draw contour
        SelectObject(dc, pen1);
        SelectObject(dc, br1);
        Ellipse(dc, t_Neuron.m_x - static_cast<int>(radius-1), t_Neuron.m_y - static_cast<int>(radius-1),
                t_Neuron.m_x + static_cast<int>(radius), t_Neuron.m_y + static_cast<int>(radius));
        DeleteObject(pen1);
        DeleteObject(br1);

        // Draw activation
        double act = t_Neuron.m_activation;
        double act2 = act;
        double ac = act2;
        Clamp(act2, -1, 1);
        Clamp(ac, -1, 1);

        if (ac < 0)
        {
            ac = -ac;
            NS::Scale(ac, 0, 1 , 100, 255);
            br2 = CreateSolidBrush(RGB(0 , 0 , static_cast<int>(ac)));
        }
        else
        {
            NS::Scale(ac, 0, 1 , 100, 255);
            br2 = CreateSolidBrush(RGB(ac , ac , ac ));
        }

        // Draw activation
        SelectObject(dc, br2);
        SelectObject(dc, pen2);
        double rad = act2 * radius + 1;
        rad = abs(rad);
        //if (!(rad < 0.5))
        Ellipse(dc, t_Neuron.m_x - rad + max_line_thickness/2, t_Neuron.m_y - rad + max_line_thickness/2,
                t_Neuron.m_x + rad, t_Neuron.m_y + rad);
        DeleteObject(pen2);
        DeleteObject(br2);


        // Display the node's number under the node
        char str[64];
        char type[64];
        char stype[64];
        stype[0]=0;
        type[0]=0;

        if ((t_Neuron.m_type != NS::NEAT::INPUT) && (t_Neuron.m_type != BIAS))
            switch(t_Neuron.m_activation_function_type)
            {
            case SIGNED_SIGMOID:
                sprintf(type, "S +/-");
                break;

            case UNSIGNED_SIGMOID:
                sprintf(type, "S");
                break;

            case TANH:
                sprintf(type, "T");
                break;

            case TANH_CUBIC:
                sprintf(type, "T3");
                break;

            case SIGNED_STEP:
                sprintf(type, "St +/-");
                break;

            case UNSIGNED_STEP:
                sprintf(type, "St");
                break;

            case SIGNED_GAUSS:
                sprintf(type, "G +/-");
                break;

            case UNSIGNED_GAUSS:
                sprintf(type, "G");
                break;

            case ABS:
                sprintf(type, "A");
                break;

            case SIGNED_SINE:
                sprintf(type, "Si +/-");
                break;

            case UNSIGNED_SINE:
                sprintf(type, "Si");
                break;

            case SIGNED_SQUARE:
                sprintf(type, "Sq +/-");
                break;

            case UNSIGNED_SQUARE:
                sprintf(type, "Sq");
                break;

            case LINEAR:
                sprintf(type, "L");
                break;

            default:
                sprintf(type, "S");
                break;
            }

        sprintf(str, "%s %3.2f\n", type, act	);

        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 0));
        int len = static_cast<int>(strlen(str)-1);
        len *= 6; // pixels for a char
        len /= 2;

        if (drawing_type == CPPN)
            TextOut(dc, t_Neuron.m_x-len, t_Neuron.m_y + static_cast<int>(radius) + 5, str, static_cast<int>(strlen(str)-1));
    }
}



#endif








