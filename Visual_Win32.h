#if 0

#ifndef _VISUALWIN32_H
#define _VISUALWIN32_H

/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2007, NEAT Sciences Ltd.
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Visual_Win32.h
// Description: Genome visualization routines for Microsoft Windows
///////////////////////////////////////////////////////////////////////////////
//#include "stdafx.h"
#include <windows.h>
#include "Genome.h"
#include "Random.h"
#include "Utils.h"

enum DrawingType
{
    STANDART,
    CPPN,
    CTRNN,
    SUBSTRATE
};

void Draw_Genome(NS::NEAT::Genome& genome, HDC& dc, int x, int y, int x_size, int y_size, DrawingType type, int neuron_radius, int max_line_thickness, int arrow_spike_length);

void Draw_NN(NS::NEAT::NeuralNetwork& net, HDC& dc, int xpos, int ypos, int rect_x_size, int rect_y_size, DrawingType drawing_type, int neuron_radius, int max_line_thickness, int arrow_spike_length);

#endif
#endif
