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
// File:        ESHyperNEAT.cpp
// Description: Evolvable-Substrate HyperNEAT phenotype builder.
//
// The algorithm follows Risi & Stanley, "An Enhanced Hypercube-Based Encoding for
// Evolving the Placement, Density and Connectivity of Neurons" (2012):
//   1. For every input, the CPPN's outgoing weight pattern is sampled with a
//      2^D-tree (a quadtree in 2D, an octree in 3D) that is refined where the
//      pattern has high variance (division and initialization).
//   2. The tree is pruned and connections are expressed only at points that lie
//      inside a "band" - where the weight differs enough from its neighbours
//      (pruning and extraction). The targets of these connections become hidden neurons.
//   3. Step 2 is repeated from the discovered hidden neurons IterationLevel times.
//   4. For every output, the incoming pattern is sampled the same way and only
//      connections from already discovered hidden neurons are kept.
//   5. Hidden neurons that are not on a path from an input to an output are removed.
//
// The hidden space has as many dimensions as the substrate's largest coordinate.
// Its extents are given by Parameters::Qtree_X / Qtree_Y (center) and
// Width / Height (half-size of the first two dimensions; the other dimensions use Width).
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

#include "Genome.h"
#include "Substrate.h"
#include "NeuralNetwork.h"
#include "Parameters.h"

namespace NEAT
{

namespace
{

// A node of the 2^D-tree
struct TreeNode
{
    std::vector<double> center;
    double half; // the half-size of this cell (relative, multiplied by the per-dimension extents)
    int level;
    double weight;
    std::vector< std::unique_ptr<TreeNode> > children;

    TreeNode(const std::vector<double> &a_center, double a_half, int a_level)
        : center(a_center), half(a_half), level(a_level), weight(0)
    {
    }
};

struct TempConnection
{
    std::vector<double> source;
    std::vector<double> target;
    double weight;
};

// Queries the CPPN for the weight of the connection a -> b
class CPPNQuery
{
public:
    CPPNQuery(Genome &a_Genome, const Substrate &a_Subst, const Parameters &a_Params, int a_Dims)
        : m_subst(a_Subst), m_params(a_Params), m_dims(a_Dims)
    {
        a_Genome.BuildPhenotype(m_cppn);
        m_num_inputs = a_Genome.NumInputs();
        m_num_outputs = a_Genome.NumOutputs();

        // to ensure network relaxation
        m_depth = 8;
        if (!a_Genome.HasLoops())
        {
            a_Genome.CalculateDepth();
            m_depth = std::max(1, static_cast<int>(a_Genome.GetDepth()));
        }
    }

    // returns the weight in [-1 .. 1] (for signed CPPN outputs) and whether it is expressed (LEO)
    double Query(const std::vector<double> &a_from, const std::vector<double> &a_to, bool &a_expressed)
    {
        std::vector<double> t_inputs(m_num_inputs, 0.0);
        for (int n = 0; n < m_dims; n++)
        {
            if (n < static_cast<int>(a_from.size())) t_inputs[n] = a_from[n];
            if (n < static_cast<int>(a_to.size())) t_inputs[m_dims + n] = a_to[n];
        }

        if (m_subst.m_with_distance)
        {
            double sum = 0;
            for (int n = 0; n < m_dims; n++)
            {
                double d = t_inputs[n] - t_inputs[m_dims + n];
                sum += d * d;
            }
            t_inputs[m_num_inputs - 2] = std::sqrt(sum);
        }

        t_inputs[m_num_inputs - 1] = m_params.CPPN_Bias;

        m_cppn.Flush();
        m_cppn.Input(t_inputs);
        for (int d = 0; d < m_depth; d++)
        {
            m_cppn.Activate();
        }

        std::vector<double> t_out = m_cppn.Output();
        a_expressed = true;
        if (m_params.Leo && (t_out.size() > 1))
        {
            a_expressed = (t_out[1] > m_params.LeoThreshold);
        }
        return t_out[0];
    }

private:
    NeuralNetwork m_cppn;
    const Substrate &m_subst;
    const Parameters &m_params;
    int m_dims;
    int m_num_inputs;
    int m_num_outputs;
    int m_depth;
};


class ESBuilder
{
public:
    ESBuilder(CPPNQuery &a_Query, const Parameters &a_Params, int a_Dims)
        : m_query(a_Query), m_params(a_Params), m_dims(a_Dims)
    {
        m_center.assign(m_dims, 0.0);
        m_extent.assign(m_dims, a_Params.Width);
        if (m_dims > 0) m_center[0] = a_Params.Qtree_X;
        if (m_dims > 1)
        {
            m_center[1] = a_Params.Qtree_Y;
            m_extent[1] = a_Params.Height;
        }
    }

    // Finds the connections going out of (or coming into) the point a_Point
    void Search(const std::vector<double> &a_Point, bool a_Outgoing, std::vector<TempConnection> &a_Connections)
    {
        std::unique_ptr<TreeNode> t_root = DivideAndInitialize(a_Point, a_Outgoing);
        PruneAndExtract(a_Point, *t_root, a_Outgoing, a_Connections);
    }

private:
    CPPNQuery &m_query;
    const Parameters &m_params;
    int m_dims;
    std::vector<double> m_center;
    std::vector<double> m_extent;

    std::vector<double> Coord(const std::vector<double> &a_rel) const
    {
        std::vector<double> c(m_dims);
        for (int n = 0; n < m_dims; n++)
        {
            c[n] = m_center[n] + a_rel[n] * m_extent[n];
        }
        return c;
    }

    double Weight(const std::vector<double> &a_Point, const std::vector<double> &a_Other, bool a_Outgoing, bool &a_Expressed)
    {
        if (a_Outgoing)
        {
            return m_query.Query(a_Point, a_Other, a_Expressed);
        }
        else
        {
            return m_query.Query(a_Other, a_Point, a_Expressed);
        }
    }

    static double Variance(const std::vector<double> &a_Vals)
    {
        if (a_Vals.empty())
        {
            return 0;
        }
        double mean = 0;
        for (double v : a_Vals) mean += v;
        mean /= a_Vals.size();
        double var = 0;
        for (double v : a_Vals) var += (v - mean) * (v - mean);
        return var / a_Vals.size();
    }

    static void CollectLeaves(const TreeNode &a_Node, std::vector<double> &a_Vals)
    {
        if (a_Node.children.empty())
        {
            a_Vals.push_back(a_Node.weight);
            return;
        }
        for (const auto &c : a_Node.children)
        {
            CollectLeaves(*c, a_Vals);
        }
    }

    static double SubtreeVariance(const TreeNode &a_Node)
    {
        std::vector<double> vals;
        CollectLeaves(a_Node, vals);
        return Variance(vals);
    }

    std::unique_ptr<TreeNode> DivideAndInitialize(const std::vector<double> &a_Point, bool a_Outgoing)
    {
        std::unique_ptr<TreeNode> t_root(new TreeNode(std::vector<double>(m_dims, 0.0), 1.0, 1));
        std::queue<TreeNode *> t_queue;
        t_queue.push(t_root.get());

        const int t_num_children = 1 << m_dims;

        while (!t_queue.empty())
        {
            TreeNode *p = t_queue.front();
            t_queue.pop();

            double t_child_half = p->half / 2.0;
            std::vector<double> t_child_weights;
            for (int k = 0; k < t_num_children; k++)
            {
                std::vector<double> c = p->center;
                for (int n = 0; n < m_dims; n++)
                {
                    c[n] += ((k >> n) & 1) ? t_child_half : -t_child_half;
                }
                std::unique_ptr<TreeNode> t_child(new TreeNode(c, t_child_half, p->level + 1));
                bool t_expressed;
                t_child->weight = Weight(a_Point, Coord(c), a_Outgoing, t_expressed);
                t_child_weights.push_back(t_child->weight);
                p->children.push_back(std::move(t_child));
            }

            // keep dividing up to the initial resolution, and further where the pattern is complex
            if ((p->level < static_cast<int>(m_params.InitialDepth)) ||
                ((p->level < static_cast<int>(m_params.MaxDepth)) && (Variance(t_child_weights) > m_params.DivisionThreshold)))
            {
                for (auto &c : p->children)
                {
                    t_queue.push(c.get());
                }
            }
        }

        return t_root;
    }

    void PruneAndExtract(const std::vector<double> &a_Point, TreeNode &a_Node, bool a_Outgoing,
                         std::vector<TempConnection> &a_Connections)
    {
        for (auto &cp : a_Node.children)
        {
            TreeNode &c = *cp;

            if (!c.children.empty() && (SubtreeVariance(c) >= m_params.VarianceThreshold))
            {
                // there is still interesting information further down
                PruneAndExtract(a_Point, c, a_Outgoing, a_Connections);
                continue;
            }

            // band pruning - express the connection only if the point differs enough
            // from its neighbours along at least one dimension
            double t_band = 0;
            bool t_expressed = true;
            for (int n = 0; n < m_dims; n++)
            {
                std::vector<double> lo = c.center, hi = c.center;
                lo[n] -= 2.0 * c.half;
                hi[n] += 2.0 * c.half;
                bool e;
                double d_lo = std::fabs(c.weight - Weight(a_Point, Coord(lo), a_Outgoing, e));
                double d_hi = std::fabs(c.weight - Weight(a_Point, Coord(hi), a_Outgoing, e));
                t_band = std::max(t_band, std::min(d_lo, d_hi));
            }
            Weight(a_Point, Coord(c.center), a_Outgoing, t_expressed);

            if ((t_band > m_params.BandThreshold) && t_expressed)
            {
                TempConnection tc;
                if (a_Outgoing)
                {
                    tc.source = a_Point;
                    tc.target = Coord(c.center);
                }
                else
                {
                    tc.source = Coord(c.center);
                    tc.target = a_Point;
                }
                tc.weight = c.weight;
                a_Connections.push_back(tc);
            }
        }
    }
};

std::vector<double> Padded(const std::vector<double> &a_Coord, int a_Dims)
{
    std::vector<double> c(a_Dims, 0.0);
    for (int n = 0; (n < a_Dims) && (n < static_cast<int>(a_Coord.size())); n++)
    {
        c[n] = a_Coord[n];
    }
    return c;
}

} // anonymous namespace


void Genome::BuildESHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst, Parameters &params)
{
    if (subst.m_input_coords.empty() || subst.m_output_coords.empty())
    {
        throw std::invalid_argument("ES-HyperNEAT needs a substrate with at least one input and one output");
    }

    const int t_dims = subst.GetMaxDims();
    if (t_dims < 1)
    {
        throw std::invalid_argument("ES-HyperNEAT needs substrate coordinates with at least one dimension");
    }
    if (t_dims > 6)
    {
        throw std::invalid_argument("ES-HyperNEAT supports substrates with up to 6 dimensions");
    }
    if (static_cast<int>(NumInputs()) < subst.GetMinCPPNInputs())
    {
        throw std::invalid_argument("The CPPN genome has fewer inputs than the substrate requires (see Substrate.GetMinCPPNInputs())");
    }
    if (NumOutputs() < 1)
    {
        throw std::invalid_argument("The CPPN genome needs at least one output");
    }

    CPPNQuery t_query(*this, subst, params, t_dims);
    ESBuilder t_builder(t_query, params, t_dims);

    std::vector< std::vector<double> > t_inputs, t_outputs;
    for (const auto &c : subst.m_input_coords) t_inputs.push_back(Padded(c, t_dims));
    for (const auto &c : subst.m_output_coords) t_outputs.push_back(Padded(c, t_dims));

    // the discovered hidden neurons, in order of discovery
    std::map<std::vector<double>, int> t_hidden_index;
    std::vector< std::vector<double> > t_hidden;

    // all candidate connections, by coordinates
    std::vector<TempConnection> t_conns;

    auto add_hidden = [&](const std::vector<double> &a_c) -> bool
    {
        if (t_hidden_index.count(a_c) == 0)
        {
            t_hidden_index[a_c] = static_cast<int>(t_hidden.size());
            t_hidden.push_back(a_c);
            return true;
        }
        return false;
    };

    // 1. input -> hidden
    std::vector< std::vector<double> > t_unexplored;
    for (const auto &in : t_inputs)
    {
        std::vector<TempConnection> found;
        t_builder.Search(in, true, found);
        for (const auto &tc : found)
        {
            if (add_hidden(tc.target))
            {
                t_unexplored.push_back(tc.target);
            }
            t_conns.push_back(tc);
        }
    }

    // 2. hidden -> hidden, a few iterations deep
    for (unsigned int it = 0; it < params.IterationLevel; it++)
    {
        std::vector< std::vector<double> > t_next;
        for (const auto &h : t_unexplored)
        {
            std::vector<TempConnection> found;
            t_builder.Search(h, true, found);
            for (const auto &tc : found)
            {
                if (add_hidden(tc.target))
                {
                    t_next.push_back(tc.target);
                }
                t_conns.push_back(tc);
            }
        }
        t_unexplored.swap(t_next);
    }

    // 3. hidden -> output (only from neurons already discovered)
    for (const auto &out : t_outputs)
    {
        std::vector<TempConnection> found;
        t_builder.Search(out, false, found);
        for (const auto &tc : found)
        {
            if (t_hidden_index.count(tc.source) != 0)
            {
                t_conns.push_back(tc);
            }
        }
    }

    // Index all points. Network layout is INPUTS, OUTPUTS, HIDDEN like in BuildHyperNEATPhenotype
    const int t_num_in = static_cast<int>(t_inputs.size());
    const int t_num_out = static_cast<int>(t_outputs.size());
    std::map<std::vector<double>, int> t_in_index, t_out_index;
    for (int i = 0; i < t_num_in; i++) t_in_index[t_inputs[i]] = i;
    for (int i = 0; i < t_num_out; i++) t_out_index[t_outputs[i]] = t_num_in + i;

    struct IdxConn { int src, dst; double w; };
    std::vector<IdxConn> t_iconns;
    std::set< std::pair<int, int> > t_seen;
    for (const auto &tc : t_conns)
    {
        int s = -1, d = -1;
        auto si = t_in_index.find(tc.source);
        if (si != t_in_index.end()) s = si->second;
        else
        {
            auto sh = t_hidden_index.find(tc.source);
            if (sh != t_hidden_index.end()) s = t_num_in + t_num_out + sh->second;
        }
        auto dout = t_out_index.find(tc.target);
        if (dout != t_out_index.end()) d = dout->second;
        else
        {
            auto dh = t_hidden_index.find(tc.target);
            if (dh != t_hidden_index.end()) d = t_num_in + t_num_out + dh->second;
        }
        if ((s < 0) || (d < 0) || (s == d))
        {
            continue;
        }
        if (t_seen.insert(std::make_pair(s, d)).second)
        {
            t_iconns.push_back({ s, d, tc.weight });
        }
    }

    // 4. keep only hidden neurons that are on a path from an input to an output
    const int t_total = t_num_in + t_num_out + static_cast<int>(t_hidden.size());
    std::vector< std::vector<int> > t_fwd(t_total), t_bwd(t_total);
    for (const auto &c : t_iconns)
    {
        t_fwd[c.src].push_back(c.dst);
        t_bwd[c.dst].push_back(c.src);
    }
    auto reach = [&](const std::vector< std::vector<int> > &a_adj, int a_from, int a_to) {
        std::vector<bool> t_mark(t_total, false);
        std::vector<int> t_stack;
        for (int i = a_from; i < a_to; i++)
        {
            t_mark[i] = true;
            t_stack.push_back(i);
        }
        while (!t_stack.empty())
        {
            int n = t_stack.back();
            t_stack.pop_back();
            for (int m : a_adj[n])
            {
                if (!t_mark[m])
                {
                    t_mark[m] = true;
                    t_stack.push_back(m);
                }
            }
        }
        return t_mark;
    };
    std::vector<bool> t_from_inputs = reach(t_fwd, 0, t_num_in);
    std::vector<bool> t_to_outputs = reach(t_bwd, t_num_in, t_num_in + t_num_out);

    std::vector<int> t_new_index(t_total, -1);
    for (int i = 0; i < t_num_in + t_num_out; i++)
    {
        t_new_index[i] = i;
    }
    int t_next_idx = t_num_in + t_num_out;
    for (int i = t_num_in + t_num_out; i < t_total; i++)
    {
        if (t_from_inputs[i] && t_to_outputs[i])
        {
            t_new_index[i] = t_next_idx++;
        }
    }

    // Build the network
    net.Clear();
    net.SetInputOutputDimentions(t_num_in, t_num_out);

    for (int i = 0; i < t_num_in; i++)
    {
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = subst.m_input_coords[i];
        t_n.m_activation_function_type = NEAT::LINEAR;
        t_n.m_type = NEAT::INPUT;
        t_n.m_split_y = 0.0;
        net.AddNeuron(t_n);
    }
    for (int i = 0; i < t_num_out; i++)
    {
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = subst.m_output_coords[i];
        t_n.m_activation_function_type = subst.m_output_nodes_activation;
        t_n.m_type = NEAT::OUTPUT;
        t_n.m_split_y = 1.0;
        net.AddNeuron(t_n);
    }
    for (int i = t_num_in + t_num_out; i < t_total; i++)
    {
        if (t_new_index[i] < 0)
        {
            continue;
        }
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = t_hidden[i - t_num_in - t_num_out];
        t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
        t_n.m_type = NEAT::HIDDEN;
        t_n.m_split_y = 0.5;
        net.AddNeuron(t_n);
    }

    for (const auto &c : t_iconns)
    {
        int s = t_new_index[c.src];
        int d = t_new_index[c.dst];
        if ((s < 0) || (d < 0))
        {
            continue;
        }
        Connection t_c;
        t_c.m_source_neuron_idx = s;
        t_c.m_target_neuron_idx = d;
        t_c.m_weight = c.w * subst.m_max_weight_and_bias;
        t_c.m_recur_flag = false;
        net.AddConnection(t_c);
    }

    net.Flush();
}

} // namespace NEAT
