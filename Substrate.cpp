////////////////////////
// Peter Chervenski
// spookey@abv.bg
////////////////////////



#include <vector>
#include "NeuralNetwork.h"
//#include "Math_Vectors.h"
#include "Utils.h"
#include "Substrate.h"

using namespace std;

#define SUBSTRATE_SPACE_SCALAR 1.0

#define PARRALEL_MATRIX_NUM_LAYERS 15

#define MIN_ACTIVATION_FOR_LINK_TRESHOLD 0.2
#define MAX_LINK_WEIGHT 8.0

namespace NEAT
{

Substrate::Substrate(substrate_config config, int inp, int hid, int outp)
{
    const double PI = 3.14152;
#if 0
    int i;
    vector3D tmp;

    num_inputs  = inp;
    num_hidden  = hid;
    num_outputs = outp;

    inputs.clear();
    hidden.clear();
    outputs.clear();

    if (config == RANDOM)
    {
        // Random positions for all nodes
        // The substrate limits are [-1 .. 1] on both axises
    	RNG t_RNG;
    	t_RNG.TimeSeed();

        // create inputs positions
        for(i=0; i<inp; i++)
        {
            tmp.x = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.y = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.z = (t_RNG.RandFloat() * 2.0) - 1.0;

            inputs.push_back( tmp );
        }

        // create hidden positions
        for(i=0; i<hid; i++)
        {
            tmp.x = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.y = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.z = (t_RNG.RandFloat() * 2.0) - 1.0;

            hidden.push_back( tmp );
        }

        // create output positions
        for(i=0; i<outp; i++)
        {
            tmp.x = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.y = (t_RNG.RandFloat() * 2.0) - 1.0; // scale to -1 .. 1
            tmp.z = (t_RNG.RandFloat() * 2.0) - 1.0;

            outputs.push_back( tmp );
        }

        return;
    }

    if (config == PARRALEL)
    {
        double xxpos=0;

        // calculate X positions for nodes

        xxpos = (2.0 / (1.0 + num_inputs));
        for(i=0; i<num_inputs; i++)
        {
            tmp.x = (xxpos + i*(2.0/(2.0 + num_inputs))) - 1.0;
            tmp.y = -1;
            tmp.z = -1;

            inputs.push_back(tmp);
        }

        xxpos = (2.0 / (1.0 + num_hidden));
        for(i=0; i<num_hidden; i++)
        {
            tmp.x = (xxpos + i*(2.0/(2.0 + num_hidden))) - 1.0;
            tmp.y = 0;
            tmp.z = 0;

            hidden.push_back(tmp);
        }

        xxpos = (2.0 / (1.0 + num_outputs));
        for(i=0; i<num_outputs; i++)
        {
            tmp.x = (xxpos + i*(2.0/(2.0 + num_outputs))) - 1.0;
            tmp.y = 1;
            tmp.z = 1;

            outputs.push_back(tmp);
        }

        return;
    }

    if (config == CIRCULAR)
    {
        double ang;
        int cur_node=0;
        int stop_node, nodes_in_ring;

        // inputs
        for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(num_inputs), cur_node++)
        {
            if (cur_node == num_inputs)
                break;

            tmp.x = sin(ang) * 0.999; // ring for the inputs
            tmp.y = cos(ang) * 0.999;
            tmp.z = -1;

            inputs.push_back( tmp );
        }

        // hidden - 2 rings!
        cur_node=0;

        /*		stop_node = num_hidden/2;
        		nodes_in_ring = num_hidden/2;
        		for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(nodes_in_ring), cur_node++)
        		{
        			if (cur_node == stop_node)
        				break;

        			tmp.x = sin(ang) * 0.2; // inner hidden ring
        			tmp.y = cos(ang) * 0.2;
        			tmp.z = 0;

        			hidden.push_back( tmp );
        		}
        */		// second ring of hidden neurons
        stop_node = num_hidden;
        nodes_in_ring = num_hidden;
        for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(nodes_in_ring), cur_node++)
        {
            if (cur_node == stop_node)
                break;

            tmp.x = sin(ang) * 0.5; // outer hidden ring
            tmp.y = cos(ang) * 0.5;
            tmp.z = 0;

            hidden.push_back( tmp );
        }


        // outputs
        cur_node=0;
        for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(num_outputs), cur_node++)
        {
            if (cur_node == num_outputs)
                break;

            tmp.x = sin(ang) * 0.2; // inner ring is for the outputs
            tmp.y = cos(ang) * 0.2;
            tmp.z = 1;

            outputs.push_back( tmp );
        }
    }

    // arrange inputs in a matrix in [-1 .. 1] and the hiddens and outputs circular around
    if (config == PARRALEL_MATRIX)
    {
        double xxpos=0;
        double yypos=0;
        double cur_layer=0;
        double layers = PARRALEL_MATRIX_NUM_LAYERS;        // num_inputs should be divisible by 2

        double inputs_per_layer = (num_inputs-1) / layers; // last input is a bias and placed elsewhere
//		double inputs_per_layer = (num_inputs) / layers;   // no bias

        double ang;
        double half_sq_x = (2.0 / inputs_per_layer) / 2.0;
        double half_sq_y = (2.0 / layers) / 2.0;
        int cur_node=0;

        yypos = -1;
        for(cur_layer=0; cur_layer<layers; cur_layer++)
        {
            // calculate X positions for nodes
            xxpos = -1;//(2.0 / (1.0 + inputs_per_layer));
            for(i=0; i<inputs_per_layer; i++)
            {
                // stretch from -1 to 1
                tmp.x = xxpos + half_sq_x;
                tmp.y = yypos + half_sq_y;
                tmp.z = -1;

                //Scale(tmp.x, -1, 1, -0.5, 0.5);
                //Scale(tmp.y, -1, 1, -0.5, 0.5);
                cur_node++;

                xxpos += 2.0/inputs_per_layer;
                inputs.push_back(tmp);
            }
            yypos += 2.0/layers;
        }

        // the bias is located at the center and between the hidden and output neurons
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0.5;
        inputs.push_back(tmp);

        // hidden
        cur_node=0;
        for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(num_hidden), cur_node++)
        {
            if (cur_node == num_hidden)
                break;

            if (num_hidden > 1)
            {
                tmp.x = sin(ang) * 0.6; // middle ring is for the hidden
                tmp.y = cos(ang) * 0.6;
                tmp.z = 0;
            }
            else
            {
                tmp.x = 0;
                tmp.y = 0;
                tmp.z = 0;
            }

            hidden.push_back( tmp );
        }

        // outputs
        cur_node=0;
        for(ang = 0.0; ang < 2*PI+1; ang += 2.0*PI/static_cast<double>(num_outputs), cur_node++)
        {
            if (cur_node == num_outputs)
                break;

            if (num_outputs > 1)
            {
                tmp.x = sin(ang) * 0.8;
                tmp.y = cos(ang) * 0.8;
                tmp.z = 1;
            }
            else
            {
                tmp.x = 0;
                tmp.y = 0;
                tmp.z = 1;
            }

            outputs.push_back( tmp );
        }
    }
#endif
}

/*
NeuralNetwork* NEAT::create_hyper_phenotype(NeuralNetwork* CPPN, Substrate* substrate)
{
	int i,j,tmp;
	NEAT::Genome  *genome;
	NEAT::NNode   *newnode;
	NEAT::Gene    *newgene;
	NEAT::NeuralNetwork *net;

	vector<double> CPPN_input;
	double weight;

	// Create an empty genome
	genome = new Genome(0, 0, 0, 0);

	// Now start adding the nodes, reading info from the substrate
	for(i=0; i<substrate->num_inputs; i++)
	{
		newnode = new NNode(SENSOR, i, NEAT::INPUT);

		newnode->substrate_x = substrate->inputs[i].x;
		newnode->substrate_y = substrate->inputs[i].y;
		newnode->substrate_z = substrate->inputs[i].z;

		newnode->split_y = 0;

		genome->node_insert(genome->nodes, newnode);
	}

	for(i=0; i<substrate->num_hidden; i++)
	{
		newnode = new NNode(NEURON, i+substrate->num_hidden, NEAT::HIDDEN);

		newnode->substrate_x = substrate->hidden[i].x;
		newnode->substrate_y = substrate->hidden[i].y;
		newnode->substrate_z = substrate->hidden[i].z;

		newnode->split_y = 0.5;

		genome->node_insert(genome->nodes, newnode);
	}

	for(i=0; i<substrate->num_outputs; i++)
	{
		newnode = new NNode(NEURON, i+substrate->num_hidden+substrate->num_outputs, NEAT::OUTPUT);

		newnode->substrate_x = substrate->outputs[i].x;
		newnode->substrate_y = substrate->outputs[i].y;
		newnode->substrate_z = substrate->outputs[i].z;

		newnode->split_y = 1.0;

		genome->node_insert(genome->nodes, newnode);
	}

	//int depth = CPPN->max_depth(); // this causes an infinite loop sometimes
	int depth = 32; // workaround, but works

	// Now the most important part.. For every pair of nodes
	// Query the CPPN with their coordinates and create a new gene
	// for the network..
	int nodes_size = genome->nodes.size();
	for(i=0; i<nodes_size; i++)
	{
		// EXPERIMENTAL
		// Make a second query of the CPPN for the time constant and bias
		// for the second neuron
		// the inputs for the second neuron are set to (0,0) and only coordinates for the
		// first neuron are input.
		CPPN_input.clear();

		// Flush the network because it has to process each coordinate independantly
		{
			unsigned int tNumNodes = CPPN->all_nodes.size();
			for(unsigned int n = 0; n < tNumNodes; n++)
			{
				CPPN->all_nodes[n]->activation = 0.0;
			}
		}

		double distance_from_center;
		distance_from_center = sqrt(sqr(genome->nodes[i]->substrate_x) + sqr(genome->nodes[i]->substrate_y));

		CPPN_input.push_back( genome->nodes[i]->substrate_x * SUBSTRATE_SPACE_SCALAR);
		CPPN_input.push_back( genome->nodes[i]->substrate_y * SUBSTRATE_SPACE_SCALAR );
//		CPPN_input.push_back( genome->nodes[i]->substrate_z * SUBSTRATE_SPACE_SCALAR );

		CPPN_input.push_back( genome->nodes[i]->substrate_x * SUBSTRATE_SPACE_SCALAR ); // x (can make it 0.0)
		CPPN_input.push_back( genome->nodes[i]->substrate_y * SUBSTRATE_SPACE_SCALAR ); // y (can make it 0.0)
//		CPPN_input.push_back( genome->nodes[i]->substrate_z * SUBSTRATE_SPACE_SCALAR ); // z (can make it 0.0)

		CPPN_input.push_back( distance_from_center ); // used as "distance from center"
		CPPN_input.push_back( 1.0 ); // bias

		CPPN->load_sensors( CPPN_input );

		// query
		for(tmp=0;tmp<(depth);tmp++)
			CPPN->activate_normal_fast(true);

		double tc = CPPN->outputs[1]->activation;
#ifdef SIGNED_ACTIVATION
		Clamp(tc, -1, 1);
		Scale(tc, -1, 1, NEAT::min_time_constant, NEAT::max_time_constant);
#else
		Clamp(tc, 0, 1);
		Scale(tc, 0, 1, NEAT::min_time_constant, NEAT::max_time_constant);
#endif
		genome->nodes[i]->time_constant = tc;

		double b = CPPN->outputs[2]->activation;
#ifdef SIGNED_ACTIVATION
		Clamp(b, -1, 1);
//		Scale(b, -1, 1, -1, 1);
#else
		Clamp(b, 0, 1);
		Scale(b, 0, 1, -1, 1);
#endif
		genome->nodes[i]->bias = b * MAX_LINK_WEIGHT;

		for(j=0; j<nodes_size; j++)
		{
			// i is the input j is the output

			// so output to inputs is not allowed and not to consider
			if ((genome->nodes[j]->gen_node_label != NEAT::INPUT) && (genome->nodes[j]->gen_node_label != NEAT::BIAS))
			{
				// Clear
				CPPN_input.clear();

				{
					// Flush the network because it has to process each coordinate independantly
					int ns = CPPN->all_nodes.size();
					for(int n=0; n<ns; n++)
					{
						CPPN->all_nodes[n]->activation = 0.0;
					}
				}

				// Query the CPPN for the weight
				// The CPPN MUST have 7 inputs
				CPPN_input.push_back( genome->nodes[i]->substrate_x * SUBSTRATE_SPACE_SCALAR );
				CPPN_input.push_back( genome->nodes[i]->substrate_y * SUBSTRATE_SPACE_SCALAR );
//				CPPN_input.push_back( genome->nodes[i]->substrate_z * SUBSTRATE_SPACE_SCALAR );
				CPPN_input.push_back( genome->nodes[j]->substrate_x * SUBSTRATE_SPACE_SCALAR );
				CPPN_input.push_back( genome->nodes[j]->substrate_y * SUBSTRATE_SPACE_SCALAR );
//				CPPN_input.push_back( genome->nodes[j]->substrate_z * SUBSTRATE_SPACE_SCALAR );

				vector3D s,d;

				s.x = genome->nodes[i]->substrate_x;
				s.y = genome->nodes[i]->substrate_y;
				s.z = genome->nodes[i]->substrate_z;

				d.x = genome->nodes[j]->substrate_x;
				d.y = genome->nodes[j]->substrate_y;
				d.z = genome->nodes[j]->substrate_z;

				CPPN_input.push_back( s.distance_to(d) );

				CPPN_input.push_back( 1.0 ); // bias

				// query
				CPPN->load_sensors( CPPN_input );
				for(tmp=0;tmp<(depth);tmp++)
					CPPN->activate_normal_fast(true);

				// the weight is the output
				weight = CPPN->outputs[0]->activation;

#ifdef SIGNED_ACTIVATION
				Clamp(weight, -1, 1);
#else
				Clamp(weight, 0, 1);
#endif


				// Add the link between the neurons
#ifndef SIGNED_ACTIVATION
				Scale(weight, 0, 1, -1, 1);
#endif
				float tUnsignedWeight =weight;
				if (tUnsignedWeight <0)
					tUnsignedWeight *= -1;

				if (tUnsignedWeight > MIN_ACTIVATION_FOR_LINK_TRESHOLD)
				{
#ifdef SIGNED_ACTIVATION
					if (weight < 0)
						Scale(weight, -1, -MIN_ACTIVATION_FOR_LINK_TRESHOLD, -1, 0);
					else
						Scale(weight, MIN_ACTIVATION_FOR_LINK_TRESHOLD, 1, 0, 1);
#else
					Scale(weight, MIN_ACTIVATION_FOR_LINK_TRESHOLD, 1, -1, 1);
#endif

					weight *= MAX_LINK_WEIGHT;


					// Let us know if this is a recurrent
					// connection. Useful for displaying.
					bool recur = true;
					if ((genome->nodes[i]->gen_node_label == NEAT::INPUT) || (genome->nodes[i]->gen_node_label == NEAT::BIAS))
					{
						recur = false;
					}

					// Now add the gene
					newgene = new Gene(weight, genome->nodes[i], genome->nodes[j], recur, 0, 0);
					genome->add_gene(genome->genes, newgene);
				}
			}
		}
	}

	net = genome->genesis(0);

	delete genome;

	// OK, return the big network
	return net;
return NULL;
}
*/
}
