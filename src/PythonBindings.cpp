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
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

/*
 * PythonBindings.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: peter
 *
 *  The Python module MultiNEAT._MultiNEAT, built with pybind11.
 */

#ifdef PYTHON_BINDINGS

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>

#include <string>
#include <vector>

#include "NeuralNetwork.h"
#include "Genes.h"
#include "Genome.h"
#include "Population.h"
#include "Species.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Physics2D.h"
#include "Random.h"
#include "Serialization.h"
#include "Substrate.h"

namespace py = pybind11;
using namespace NEAT;

// These containers are exposed as reference types (like the old Boost.Python
// vector_indexing_suite), so code such as pop.Species[0].Individuals[0].SetFitness(1)
// modifies the population in place instead of a copy.
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector<double> >);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector<float> >);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector<int> >);
PYBIND11_MAKE_OPAQUE(std::vector<Genome>);
PYBIND11_MAKE_OPAQUE(std::vector<Species>);
PYBIND11_MAKE_OPAQUE(std::vector<Neuron>);
PYBIND11_MAKE_OPAQUE(std::vector<Connection>);
PYBIND11_MAKE_OPAQUE(std::vector<NeuronGene>);
PYBIND11_MAKE_OPAQUE(std::vector<LinkGene>);
PYBIND11_MAKE_OPAQUE(std::vector<PhenotypeBehavior>);

namespace
{

// Adds pickle support based on the class' serialize() method
template<class T, class PyClass>
void AddPickling(PyClass &a_class)
{
    a_class.def(py::pickle(
        [](const T &a_obj)
        {
            return py::bytes(SerializeToString(a_obj));
        },
        [](const py::bytes &a_state)
        {
            T t_obj;
            DeserializeFromString(static_cast<std::string>(a_state), t_obj);
            return t_obj;
        }));
}

// Lets Python classes derive from PhenotypeBehavior and override its methods
class PyPhenotypeBehavior : public PhenotypeBehavior
{
public:
    using PhenotypeBehavior::PhenotypeBehavior;

    bool Acquire(Genome *a_Genome) override
    {
        PYBIND11_OVERRIDE(bool, PhenotypeBehavior, Acquire, a_Genome);
    }

    double Distance_To(PhenotypeBehavior *a_Other) override
    {
        PYBIND11_OVERRIDE(double, PhenotypeBehavior, Distance_To, a_Other);
    }

    bool Successful() override
    {
        PYBIND11_OVERRIDE(bool, PhenotypeBehavior, Successful, );
    }
};

// The constructor used by most of the examples and old code:
// Genome(ID, NumInputs, NumHidden, NumOutputs, FS_NEAT, OutputActType, HiddenActType,
//        SeedType, Parameters, NumLayers, FS_NEAT_links=1)
Genome MakeGenomeLegacy(int a_ID, int a_NumInputs, int a_NumHidden, int a_NumOutputs, bool a_FS_NEAT,
                        ActivationFunction a_OutputActType, ActivationFunction a_HiddenActType,
                        int a_SeedType, const Parameters &a_Parameters, int a_NumLayers, int a_FS_NEAT_links)
{
    GenomeInitStruct in;
    in.NumInputs = a_NumInputs;
    in.NumHidden = a_NumHidden;
    in.NumOutputs = a_NumOutputs;
    in.FS_NEAT = a_FS_NEAT;
    in.OutputActType = a_OutputActType;
    in.HiddenActType = a_HiddenActType;
    in.SeedType = static_cast<GenomeSeedType>(a_SeedType);
    in.NumLayers = a_NumLayers;
    in.FS_NEAT_links = a_FS_NEAT_links;
    if ((in.SeedType == LAYERED) && (in.NumLayers < 1) && (in.NumHidden > 0))
    {
        in.NumLayers = 1;
    }
    Genome g(a_Parameters, in);
    g.SetID(a_ID);
    return g;
}

// A read/write property for a py::object member that returns None while the member is empty
template<class C>
auto ObjectGetter(py::object C::*a_member)
{
    return [a_member](const C &c) -> py::object
    {
        const py::object &o = c.*a_member;
        return o ? o : py::none();
    };
}

template<class C>
auto ObjectSetter(py::object C::*a_member)
{
    return [a_member](C &c, py::object a_value) { c.*a_member = a_value; };
}

template<class Vec>
void BindVector(py::module_ &m, const char *a_name)
{
    py::bind_vector<Vec>(m, a_name);
    py::implicitly_convertible<py::list, Vec>();
    py::implicitly_convertible<py::tuple, Vec>();
}

// Adds bodies and shapes (or iterables of them) to / removes them from a space
void SpaceAddOrRemove(Physics2D::Space &a_space, py::handle a_obj, bool a_add)
{
    if (py::isinstance<Physics2D::Body>(a_obj))
    {
        auto b = a_obj.cast< std::shared_ptr<Physics2D::Body> >();
        if (a_add) a_space.AddBody(b); else a_space.RemoveBody(b);
    }
    else if (py::isinstance<Physics2D::Shape>(a_obj))
    {
        auto sh = a_obj.cast< std::shared_ptr<Physics2D::Shape> >();
        if (a_add) a_space.AddShape(sh); else a_space.RemoveShape(sh);
    }
    else if (py::isinstance<py::iterable>(a_obj))
    {
        for (auto item : a_obj)
        {
            SpaceAddOrRemove(a_space, item, a_add);
        }
    }
    else
    {
        throw py::type_error("Space.add/remove accepts Body and Shape objects (or lists of them)");
    }
}

void BindPhysics2D(py::module_ &m)
{
    using namespace NEAT::Physics2D;

    py::class_<Vec2>(m, "Vec2d", "A 2D vector")
            .def(py::init<>())
            .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
            .def(py::init([](py::sequence a_s)
                 {
                     if (py::len(a_s) != 2) throw py::value_error("Vec2d needs 2 values");
                     return Vec2(a_s[0].cast<double>(), a_s[1].cast<double>());
                 }), py::arg("xy"))
            .def_readwrite("x", &Vec2::x)
            .def_readwrite("y", &Vec2::y)
            .def("length", &Vec2::Length)
            .def("dot", &Vec2::Dot, py::arg("other"))
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * double())
            .def(py::self / double())
            .def(-py::self)
            .def(py::self == py::self)
            .def("__len__", [](const Vec2 &) { return 2; })
            .def("__getitem__", [](const Vec2 &v, int i)
                 {
                     if (i == 0 || i == -2) return v.x;
                     if (i == 1 || i == -1) return v.y;
                     throw py::index_error();
                 })
            .def("__iter__", [](const Vec2 &v) { return py::iter(py::make_tuple(v.x, v.y)); })
            .def("__repr__", [](const Vec2 &v) { return "Vec2d(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")"; })
            .def(py::pickle([](const Vec2 &v) { return py::make_tuple(v.x, v.y); },
                            [](py::tuple t) { return Vec2(t[0].cast<double>(), t[1].cast<double>()); }))
            ;
    py::implicitly_convertible<py::tuple, Vec2>();
    py::implicitly_convertible<py::list, Vec2>();

    py::class_<Body, std::shared_ptr<Body> > body(m, "Body", "A rigid body without rotation");
    body.def(py::init<double>(), py::arg("mass") = 1.0, "A dynamic body with the given mass")
            .def_static("static", &Body::Static, "An immovable body, for walls and floors")
            .def_property_readonly("is_static", [](const Body &b) { return b.type == Body::STATIC; })
            .def_readwrite("mass", &Body::mass)
            .def_readwrite("position", &Body::position)
            .def_readwrite("velocity", &Body::velocity)
            .def_readwrite("force", &Body::force)
            .def_readwrite("velocity_limit", &Body::velocity_limit)
            .def("apply_impulse", &Body::ApplyImpulse, py::arg("impulse"), "Changes the velocity by impulse / mass")
            ;

    py::class_<Shape, std::shared_ptr<Shape> >(m, "Shape", "A circle or a thick line segment attached to a body")
            .def_property_readonly("is_circle", [](const Shape &s) { return s.kind == Shape::CIRCLE; })
            .def_property_readonly("is_segment", [](const Shape &s) { return s.kind == Shape::SEGMENT; })
            .def_readonly("body", &Shape::body)
            .def_readwrite("radius", &Shape::radius)
            .def_readwrite("offset", &Shape::offset)
            .def_readwrite("a", &Shape::a)
            .def_readwrite("b", &Shape::b)
            .def_readwrite("elasticity", &Shape::elasticity)
            .def_readwrite("friction", &Shape::friction)
            .def_readwrite("collision_type", &Shape::collision_type)
            .def_readwrite("sensor", &Shape::sensor)
            .def_property_readonly("world_center", &Shape::WorldCenter)
            .def_property_readonly("world_a", &Shape::WorldA)
            .def_property_readonly("world_b", &Shape::WorldB)
            ;

    m.def("Circle", &MakeCircle, py::arg("body"), py::arg("radius"), py::arg("offset") = Vec2(),
          "Creates a circle shape attached to the body");
    m.def("Segment", &MakeSegment, py::arg("body"), py::arg("a"), py::arg("b"), py::arg("radius"),
          "Creates a thick line segment (capsule) attached to the body");

    py::class_<Space>(m, "Space", "The simulated world")
            .def(py::init<>())
            .def_readwrite("gravity", &Space::gravity)
            .def_readwrite("iterations", &Space::iterations)
            .def_readwrite("slop", &Space::slop)
            .def("add", [](Space &s, py::args a_objs) { SpaceAddOrRemove(s, a_objs, true); },
                 "Adds bodies and shapes (or lists of them)")
            .def("remove", [](Space &s, py::args a_objs) { SpaceAddOrRemove(s, a_objs, false); },
                 "Removes bodies and shapes (or lists of them)")
            .def("step", &Space::Step, py::arg("dt"))
            .def("add_collision_handler", &Space::AddCollisionHandler,
                 py::arg("type_a"), py::arg("type_b"), py::arg("begin") = CollisionCallback(),
                 py::arg("separate") = CollisionCallback(),
                 "Calls begin(shape_a, shape_b) / separate(shape_a, shape_b) when shapes with these "
                 "collision types start / stop touching")
            .def("remove_collision_handler", &Space::RemoveCollisionHandler, py::arg("type_a"), py::arg("type_b"))
            .def("touching", &Space::Touching, py::arg("shape_a"), py::arg("shape_b"), "True if the two shapes touched in the last step")
            .def_property_readonly("bodies", [](const Space &s)
                 {
                     py::list l;
                     for (const auto &b : s.Bodies()) l.append(b);
                     return l;
                 })
            .def_property_readonly("shapes", [](const Space &s)
                 {
                     py::list l;
                     for (const auto &sh : s.Shapes()) l.append(sh);
                     return l;
                 })
            ;
}

} // anonymous namespace


PYBIND11_MODULE(_MultiNEAT, m)
{
    m.doc() = "MultiNEAT - NeuroEvolution of Augmenting Topologies (NEAT, HyperNEAT, ES-HyperNEAT, novelty search)";

    py::module_ physics = m.def_submodule("_physics2d", "A small 2D physics engine for simulated tasks");
    BindPhysics2D(physics);

/////////////////////////////////////////////////////////
// Containers
/////////////////////////////////////////////////////////

    py::bind_vector< std::vector<double> >(m, "DoublesList", py::buffer_protocol());
    py::implicitly_convertible<py::list, std::vector<double> >();
    py::implicitly_convertible<py::tuple, std::vector<double> >();
    BindVector< std::vector< std::vector<double> > >(m, "DoublesList2D");
    py::bind_vector< std::vector<float> >(m, "FloatsList", py::buffer_protocol());
    py::implicitly_convertible<py::list, std::vector<float> >();
    py::implicitly_convertible<py::tuple, std::vector<float> >();
    BindVector< std::vector< std::vector<float> > >(m, "FloatsList2D");
    py::bind_vector< std::vector<int> >(m, "IntsList", py::buffer_protocol());
    py::implicitly_convertible<py::list, std::vector<int> >();
    py::implicitly_convertible<py::tuple, std::vector<int> >();
    BindVector< std::vector< std::vector<int> > >(m, "IntsList2D");

    // all classes are declared first, so the signatures in the docstrings use the Python names
    py::class_<RNG> rng(m, "RNG", "A random number generator (Mersenne Twister)");
    py::class_<Connection> connection(m, "Connection", "A connection of a NeuralNetwork");
    py::class_<Neuron> neuron(m, "Neuron", "A neuron of a NeuralNetwork");
    py::class_<NeuralNetwork> network(m, "NeuralNetwork", "A neural network - the phenotype built from a genome");
    py::class_<LinkGene> linkgene(m, "LinkGene", "A link (connection) gene");
    py::class_<NeuronGene> neurongene(m, "NeuronGene", "A neuron gene");
    py::class_<GenomeInitStruct> initstruct(m, "GenomeInitStruct", "Describes the structure of a seed genome");
    py::class_<Genome> genome(m, "Genome", "A genome - the genetic description of a neural network");
    py::class_<Species> species(m, "Species", "A group of similar genomes");
    py::class_<Substrate> substrate(m, "Substrate", "The geometry of a HyperNEAT network - the coordinates of its neurons");
    py::class_<PhenotypeBehavior, PyPhenotypeBehavior> behavior(m, "PhenotypeBehavior", "Base class of behaviors for the C++ novelty search (can be subclassed in Python)");
    py::class_<Population> population(m, "Population", "A population of genomes divided into species - the evolutionary algorithm");
    py::class_<Parameters> parameters(m, "Parameters", "All the settings of the algorithm");

///////////////////////////////////////////////////////////////////
// Enums
///////////////////////////////////////////////////////////////////

    py::enum_<NeuronType>(m, "NeuronType", py::arithmetic())
        .value("NONE", NONE)
        .value("INPUT", INPUT)
        .value("BIAS", BIAS)
        .value("HIDDEN", HIDDEN)
        .value("OUTPUT", OUTPUT)
        ;

    py::enum_<ActivationFunction>(m, "ActivationFunction", py::arithmetic())
        .value("SIGNED_SIGMOID", SIGNED_SIGMOID)
        .value("UNSIGNED_SIGMOID", UNSIGNED_SIGMOID)
        .value("TANH", TANH)
        .value("TANH_CUBIC", TANH_CUBIC)
        .value("SIGNED_STEP", SIGNED_STEP)
        .value("UNSIGNED_STEP", UNSIGNED_STEP)
        .value("SIGNED_GAUSS", SIGNED_GAUSS)
        .value("UNSIGNED_GAUSS", UNSIGNED_GAUSS)
        .value("ABS", ABS)
        .value("SIGNED_SINE", SIGNED_SINE)
        .value("UNSIGNED_SINE", UNSIGNED_SINE)
        .value("LINEAR", LINEAR)
        .value("RELU", RELU)
        .value("SOFTPLUS", SOFTPLUS)
        ;

    py::enum_<SearchMode>(m, "SearchMode", py::arithmetic())
        .value("COMPLEXIFYING", COMPLEXIFYING)
        .value("SIMPLIFYING", SIMPLIFYING)
        .value("BLENDED", BLENDED)
        ;

    py::enum_<GenomeSeedType>(m, "GenomeSeedType", py::arithmetic())
        .value("PERCEPTRON", PERCEPTRON)
        .value("LAYERED", LAYERED)
        ;

///////////////////////////////////////////////////////////////////
// RNG class
///////////////////////////////////////////////////////////////////

    rng
            .def(py::init<>())
            .def("Seed", &RNG::Seed, py::arg("seed"))
            .def("TimeSeed", &RNG::TimeSeed, "Seeds from a hardware random source and the clock.")
            .def("RandPosNeg", &RNG::RandPosNeg, "Randomly 1 or -1.")
            .def("RandInt", &RNG::RandInt, py::arg("low"), py::arg("high"), "A random integer in [low, high].")
            .def("RandFloat", &RNG::RandFloat, "A random number in [0, 1).")
            .def("RandFloatSigned", &RNG::RandFloatSigned, "A random number in (-1, 1), small values are more likely.")
            .def("RandGaussSigned", &RNG::RandGaussSigned, "A normally distributed number clamped to [-1, 1].")
            .def("Roulette", [](RNG &r, std::vector<double> a_probs) { return r.Roulette(a_probs); }, py::arg("weights"), "A random index chosen with probabilities proportional to the weights.")
            ;
    AddPickling<RNG>(rng);

///////////////////////////////////////////////////////////////////
// Neural Network class
///////////////////////////////////////////////////////////////////

    connection
            .def(py::init<>())
            .def_readwrite("source_neuron_idx", &Connection::m_source_neuron_idx)
            .def_readwrite("target_neuron_idx", &Connection::m_target_neuron_idx)
            .def_readwrite("weight", &Connection::m_weight)
            .def_readwrite("recur_flag", &Connection::m_recur_flag)
            .def_readwrite("hebb_rate", &Connection::m_hebb_rate)
            .def_readwrite("hebb_pre_rate", &Connection::m_hebb_pre_rate)
            .def(py::self == py::self)
            ;

    neuron
            .def(py::init<>())
            .def_readwrite("a", &Neuron::m_a)
            .def_readwrite("b", &Neuron::m_b)
            .def_readwrite("time_const", &Neuron::m_timeconst)
            .def_readwrite("bias", &Neuron::m_bias)
            .def_readwrite("activation", &Neuron::m_activation)
            .def_readwrite("activation_function_type", &Neuron::m_activation_function_type)
            .def_readwrite("split_y", &Neuron::m_split_y)
            .def_readwrite("type", &Neuron::m_type)
            .def_readwrite("x", &Neuron::m_x)
            .def_readwrite("y", &Neuron::m_y)
            .def_readwrite("z", &Neuron::m_z)
            .def_readwrite("substrate_coords", &Neuron::m_substrate_coords)
            .def(py::self == py::self)
            ;

    py::bind_vector< std::vector<Neuron> >(m, "NeuronList");
    py::bind_vector< std::vector<Connection> >(m, "ConnectionList");

    network
            .def(py::init<>())
            .def(py::init<bool>(), py::arg("minimal"), "minimal=False creates a small XOR network for testing")

            .def("InitRTRLMatrix", &NeuralNetwork::InitRTRLMatrix, "Allocates the data for RTRL learning.")
            .def("RTRL_update_gradients", &NeuralNetwork::RTRL_update_gradients, "RTRL: updates the sensitivities after an activation.")
            .def("RTRL_update_error", &NeuralNetwork::RTRL_update_error, py::arg("target"), "RTRL: accumulates the weight changes for the target value of the first output.")
            .def("RTRL_update_weights", &NeuralNetwork::RTRL_update_weights, "RTRL: applies the accumulated weight changes.")

            .def("ActivateFast", &NeuralNetwork::ActivateFast, "Activates once, assuming unsigned sigmoid neurons everywhere.")
            .def("Activate", &NeuralNetwork::Activate, "Activates once - the signals move one connection further.")
            .def("ActivateUseInternalBias", &NeuralNetwork::ActivateUseInternalBias, "Like Activate(), but adds each neuron's bias to its input.")
            .def("ActivateLeaky", &NeuralNetwork::ActivateLeaky, py::arg("dt"), "Activates leaky integrator neurons for a time step dt.")

            .def("Adapt", &NeuralNetwork::Adapt, py::arg("params"), "Hebbian learning step.")

            .def("Flush", &NeuralNetwork::Flush, "Resets all activations.")
            .def("FlushCube", &NeuralNetwork::FlushCube, "Resets the RTRL sensitivities.")
            .def("FlushCude", &NeuralNetwork::FlushCube) // old misspelled name

            .def("NumInputs", &NeuralNetwork::NumInputs)
            .def("NumOutputs", &NeuralNetwork::NumOutputs)

            .def("Clear", &NeuralNetwork::Clear, "Removes all neurons and connections.")
            .def("Save", [](NeuralNetwork &n, const std::string &a_filename) { n.Save(a_filename.c_str()); }, py::arg("filename"), "Saves the network to a text file.")
            .def("Load", [](NeuralNetwork &n, const std::string &a_filename) { return n.Load(a_filename.c_str()); }, py::arg("filename"), "Loads a network saved with Save(). Returns False if the file is missing or malformed.")

            .def("Input", [](NeuralNetwork &n, py::object a_inputs) { n.Input_python(a_inputs); }, py::arg("inputs"),
                 "Sets the inputs. Accepts any sequence of numbers (list, tuple, numpy array).")
            .def("Output", &NeuralNetwork::Output, "The activations of the output neurons.")

            .def("AddNeuron", &NeuralNetwork::AddNeuron, py::arg("neuron"))
            .def("AddConnection", &NeuralNetwork::AddConnection, py::arg("connection"))
            .def("SetInputOutputDimentions", &NeuralNetwork::SetInputOutputDimentions, py::arg("inputs"), py::arg("outputs"))

            .def("GetTotalConnectionLength", &NeuralNetwork::GetTotalConnectionLength, "The number of connections.")

            .def_readwrite("neurons", &NeuralNetwork::m_neurons)
            .def_readwrite("connections", &NeuralNetwork::m_connections)
            ;

///////////////////////////////////////////////////////////////////
// Genome class
///////////////////////////////////////////////////////////////////

    linkgene
            .def(py::init<>())
            .def_readwrite("FromNeuronID", &LinkGene::m_FromNeuronID)
            .def_readwrite("ToNeuronID", &LinkGene::m_ToNeuronID)
            .def_readwrite("InnovationID", &LinkGene::m_InnovationID)
            .def_readwrite("Weight", &LinkGene::m_Weight)
            .def_readwrite("IsRecurrent", &LinkGene::m_IsRecurrent)
            .def(py::self == py::self)
            ;

    neurongene
            .def(py::init<>())
            .def_readwrite("ID", &NeuronGene::m_ID)
            .def_readwrite("A", &NeuronGene::m_A)
            .def_readwrite("B", &NeuronGene::m_B)
            .def_readwrite("TimeConstant", &NeuronGene::m_TimeConstant)
            .def_readwrite("Bias", &NeuronGene::m_Bias)
            .def_readwrite("ActFunction", &NeuronGene::m_ActFunction)
            .def_readwrite("Type", &NeuronGene::m_Type)
            .def_readwrite("SplitY", &NeuronGene::m_SplitY)
            .def(py::self == py::self)
            ;

    py::bind_vector< std::vector<NeuronGene> >(m, "NeuronGeneList");
    py::bind_vector< std::vector<LinkGene> >(m, "LinkGeneList");

    initstruct
            .def(py::init<>())
            .def_readwrite("NumInputs", &GenomeInitStruct::NumInputs)
            .def_readwrite("NumHidden", &GenomeInitStruct::NumHidden)
            .def_readwrite("NumOutputs", &GenomeInitStruct::NumOutputs)
            .def_readwrite("FS_NEAT", &GenomeInitStruct::FS_NEAT)
            .def_readwrite("OutputActType", &GenomeInitStruct::OutputActType)
            .def_readwrite("HiddenActType", &GenomeInitStruct::HiddenActType)
            .def_readwrite("SeedType", &GenomeInitStruct::SeedType)
            .def_readwrite("NumLayers", &GenomeInitStruct::NumLayers)
            .def_readwrite("FS_NEAT_links", &GenomeInitStruct::FS_NEAT_links)
            ;

    genome
            .def(py::init<>())
            .def(py::init([](const std::string &a_filename) { return Genome(a_filename.c_str()); }),
                 py::arg("filename"))
            .def(py::init<const Parameters &, const GenomeInitStruct &>(),
                 py::arg("params"), py::arg("init_struct"))
            .def(py::init(&MakeGenomeLegacy),
                 py::arg("ID"), py::arg("NumInputs"), py::arg("NumHidden"), py::arg("NumOutputs"),
                 py::arg("FS_NEAT"), py::arg("OutputActType"), py::arg("HiddenActType"),
                 py::arg("SeedType"), py::arg("params"), py::arg("NumLayers") = 0,
                 py::arg("FS_NEAT_links") = 1)

            .def("NumNeurons", &Genome::NumNeurons, "The number of neuron genes.")
            .def("NumLinks", &Genome::NumLinks, "The number of link genes.")
            .def("NumInputs", &Genome::NumInputs, "The number of inputs (including the bias).")
            .def("NumOutputs", &Genome::NumOutputs, "The number of outputs.")

            .def_readwrite("NeuronGenes", &Genome::m_NeuronGenes)
            .def_readwrite("LinkGenes", &Genome::m_LinkGenes)
            .def_property("behavior", ObjectGetter(&Genome::m_behavior), ObjectSetter(&Genome::m_behavior))

            .def("GetFitness", &Genome::GetFitness)
            .def("SetFitness", &Genome::SetFitness, py::arg("fitness"), "Sets the fitness - the higher the better.")
            .def("GetAdjFitness", &Genome::GetAdjFitness, "The fitness after fitness sharing.")
            .def("GetID", &Genome::GetID)
            .def("SetID", &Genome::SetID, py::arg("id"))
            .def("GetDepth", &Genome::GetDepth, "The depth computed by CalculateDepth().")
            .def("CalculateDepth", &Genome::CalculateDepth, "Computes the longest feed-forward path from the inputs to the outputs.")
            .def("DerivePhenotypicChanges", &Genome::DerivePhenotypicChanges, py::arg("net"), "Copies the weights of a network built from this genome back into the genome.")
            .def("HasLoops", &Genome::HasLoops, "True if the network has cycles.")
            .def("HasDeadEnds", &Genome::HasDeadEnds, "True if a hidden neuron has no inputs or no outputs.")

            .def("PrintAllTraits", &Genome::PrintAllTraits, "Prints the traits of the genome, its neurons and links.")
            .def("CompatibilityDistance", &Genome::CompatibilityDistance, py::arg("other"), py::arg("params"), "The compatibility distance used for speciation.")
            .def("IsCompatibleWith", &Genome::IsCompatibleWith, py::arg("other"), py::arg("params"), "True if the distance is below the compatibility threshold.")

            .def("BuildPhenotype", &Genome::BuildPhenotype, py::arg("net"), "Builds the neural network this genome encodes into net.")
            .def("BuildHyperNEATPhenotype", &Genome::BuildHyperNEATPhenotype, py::arg("net"), py::arg("substrate"), "Uses this genome as a CPPN to build a HyperNEAT network on the substrate.")
            .def("BuildESHyperNEATPhenotype", &Genome::BuildESHyperNEATPhenotype, py::arg("net"), py::arg("substrate"), py::arg("params"), "Uses this genome as a CPPN to build an ES-HyperNEAT network (the hidden neurons are discovered).")

            .def("Randomize_LinkWeights", &Genome::Randomize_LinkWeights, py::arg("params"), py::arg("rng"), "Sets all weights to random values in [MinWeight, MaxWeight].")
            .def("Randomize_Traits", &Genome::Randomize_Traits, py::arg("params"), py::arg("rng"), "Sets all traits to random values.")
            .def("Mutate_LinkWeights", &Genome::Mutate_LinkWeights, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronActivations_A", &Genome::Mutate_NeuronActivations_A, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronActivations_B", &Genome::Mutate_NeuronActivations_B, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronActivation_Type", &Genome::Mutate_NeuronActivation_Type, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronTimeConstants", &Genome::Mutate_NeuronTimeConstants, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronBiases", &Genome::Mutate_NeuronBiases, py::arg("params"), py::arg("rng"))
            .def("Mutate_NeuronTraits", &Genome::Mutate_NeuronTraits, py::arg("params"), py::arg("rng"))
            .def("Mutate_LinkTraits", &Genome::Mutate_LinkTraits, py::arg("params"), py::arg("rng"))
            .def("Mutate_GenomeTraits", &Genome::Mutate_GenomeTraits, py::arg("params"), py::arg("rng"))
            .def("GetNeuronTraits", &Genome::GetNeuronTraits, "A list of (neuron ID, type, {trait: value}).")
            .def("GetLinkTraits", &Genome::GetLinkTraits, py::arg("with_weights") = false, "A list of (from ID, to ID, {trait: value}[, weight]).")
            .def("GetGenomeTraits", &Genome::GetGenomeTraits, "The genome traits as a dict.")

            .def("SetGenomeTraits", &Genome::SetGenomeTraits, py::arg("traits"), "Replaces the genome traits (experimental).") // experimental

            .def("FailsConstraints", &Genome::FailsConstraints, py::arg("params"), "True if the genome has dead ends, no links, forbidden loops or fails CustomConstraints.")

            .def("IsEvaluated", &Genome::IsEvaluated)
            .def("SetEvaluated", &Genome::SetEvaluated, "Marks the genome as evaluated (needed for steady-state evolution).")
            .def("ResetEvaluated", &Genome::ResetEvaluated)

            .def("Save", [](Genome &g, const std::string &a_filename) { g.Save(a_filename.c_str()); }, py::arg("filename"), "Saves the genome to a text file (without traits).")

            .def(py::self == py::self)
            ;
    AddPickling<Genome>(genome);

    py::bind_vector< std::vector<Genome> >(m, "GenomeList");

///////////////////////////////////////////////////////////////////
// Species class
///////////////////////////////////////////////////////////////////

    species
            .def(py::init<>())
            .def(py::init<const Genome &, const Parameters &, int>(), py::arg("seed_genome"), py::arg("params"), py::arg("id"))
            .def("GetLeader", &Species::GetLeader, py::return_value_policy::reference_internal, "A reference to the fittest member.")
            .def("GetRepresentative", &Species::GetRepresentative, py::return_value_policy::reference_internal, "A reference to the member new genomes are compared with.")
            .def("NumIndividuals", &Species::NumIndividuals, "The number of members.")
            .def("NumEvaluated", &Species::NumEvaluated, "The number of evaluated members.")
            .def("GensNoImprovement", &Species::GensNoImprovement, "Generations without improvement of the best fitness.")
            .def("ID", &Species::ID, "The unique ID of the species.")
            .def("AgeGens", &Species::AgeGens, "The age in generations.")
            .def("AgeEvals", &Species::AgeEvals, "The age in evaluations (steady-state evolution).")
            .def("IsBestSpecies", &Species::IsBestSpecies, "True for the species with the best genome (protected from dying out).")
            .def("GetBestFitness", &Species::GetBestFitness, "The best fitness the species ever had.")
            .def("GetActualBestFitness", &Species::GetActualBestFitness, "The best fitness of the current evaluated members.")
            .def_readwrite("Individuals", &Species::m_Individuals)
            .def_readonly("Red", &Species::m_R)
            .def_readonly("Green", &Species::m_G)
            .def_readonly("Blue", &Species::m_B)
            .def_readwrite("EvalsNoImprovement", &Species::m_EvalsNoImprovement)
            .def_readwrite("BestFitness", &Species::m_BestFitness)
            .def_readonly("AverageFitness", &Species::m_AverageFitness)
            .def(py::self == py::self)
            ;
    AddPickling<Species>(species);

    py::bind_vector< std::vector<Species> >(m, "SpeciesList");

///////////////////////////////////////////////////////////////////
// Substrate class
///////////////////////////////////////////////////////////////////

    substrate
            .def(py::init<>())
            .def(py::init<py::object, py::object, py::object>(),
                 py::arg("inputs"), py::arg("hidden"), py::arg("outputs"))
            .def("SetNeurons", static_cast<void (Substrate::*)(py::object, py::object, py::object)>(&Substrate::SetNeurons), py::arg("inputs"), py::arg("hidden"), py::arg("outputs"), "Replaces the coordinates of the neurons.")
            .def("GetMinCPPNInputs", &Substrate::GetMinCPPNInputs, "The number of inputs a CPPN genome needs for this substrate.")
            .def("GetMinCPPNOutputs", &Substrate::GetMinCPPNOutputs, "The number of outputs a CPPN genome needs for this substrate.")
            .def("GetMaxDims", &Substrate::GetMaxDims, "The largest number of coordinates of any neuron.")
            .def("PrintInfo", &Substrate::PrintInfo)
            .def("SetCustomConnectivity", static_cast<void (Substrate::*)(py::object)>(&Substrate::SetCustomConnectivity), py::arg("connections"), "Query only these connections - a list of [src_type, src_index, dst_type, dst_index].")
            .def("ClearCustomConnectivity", &Substrate::ClearCustomConnectivity)

            .def_readwrite("m_leaky", &Substrate::m_leaky)
            .def_readwrite("m_with_distance", &Substrate::m_with_distance)
            .def_readwrite("m_custom_conn_obeys_flags", &Substrate::m_custom_conn_obeys_flags)
            .def_readwrite("m_query_weights_only", &Substrate::m_query_weights_only)
            .def_readwrite("m_hidden_nodes_activation", &Substrate::m_hidden_nodes_activation)
            .def_readwrite("m_output_nodes_activation", &Substrate::m_output_nodes_activation)

            .def_readwrite("m_allow_input_hidden_links", &Substrate::m_allow_input_hidden_links)
            .def_readwrite("m_allow_input_output_links", &Substrate::m_allow_input_output_links)
            .def_readwrite("m_allow_hidden_hidden_links", &Substrate::m_allow_hidden_hidden_links)
            .def_readwrite("m_allow_hidden_output_links", &Substrate::m_allow_hidden_output_links)
            .def_readwrite("m_allow_output_hidden_links", &Substrate::m_allow_output_hidden_links)
            .def_readwrite("m_allow_output_output_links", &Substrate::m_allow_output_output_links)
            .def_readwrite("m_allow_looped_hidden_links", &Substrate::m_allow_looped_hidden_links)
            .def_readwrite("m_allow_looped_output_links", &Substrate::m_allow_looped_output_links)

            .def_readwrite("m_max_weight_and_bias", &Substrate::m_max_weight_and_bias)
            .def_readwrite("m_min_time_const", &Substrate::m_min_time_const)
            .def_readwrite("m_max_time_const", &Substrate::m_max_time_const)

            .def_readwrite("m_input_coords", &Substrate::m_input_coords)
            .def_readwrite("m_hidden_coords", &Substrate::m_hidden_coords)
            .def_readwrite("m_output_coords", &Substrate::m_output_coords)
            ;
    AddPickling<Substrate>(substrate);

///////////////////////////////////////////////////////////////////
// PhenotypeBehavior class
///////////////////////////////////////////////////////////////////

    behavior
            .def(py::init<>())
            .def("Acquire", &PhenotypeBehavior::Acquire, py::arg("genome"), "Override: evaluates the genome and records its behavior. Return True if it was successful.")
            .def("Distance_To", &PhenotypeBehavior::Distance_To, py::arg("other"), "Override: the distance to another behavior.")
            .def("Successful", &PhenotypeBehavior::Successful, "Override: True if this is the behavior we are looking for.")
            .def_readwrite("m_Data", &PhenotypeBehavior::m_Data)
            .def(py::self == py::self)
            ;

    py::bind_vector< std::vector<PhenotypeBehavior> >(m, "PhenotypeBehaviorList");

///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

    population
            .def(py::init<const Genome &, const Parameters &, bool, double, int>(),
                 py::arg("seed_genome"), py::arg("params"), py::arg("randomize_weights"),
                 py::arg("randomization_range"), py::arg("rng_seed"))
            .def(py::init<const std::string>(), py::arg("filename"))
            .def("Epoch", &Population::Epoch, "Creates the next generation. All genomes must have their fitness set.")
            .def("Tick", &Population::Tick, py::return_value_policy::reference_internal, py::arg("removed_genome"), "Steady-state evolution: replaces the worst evaluated genome (copied into removed_genome) with one new baby and returns the baby. The returned genome is valid until the next Tick() or Epoch().")
            .def("InitPhenotypeBehaviorData", &Population::InitPhenotypeBehaviorData, py::arg("behaviors"), py::arg("archive"), "C++ novelty search: assigns one PhenotypeBehavior to every genome and sets the archive.")
            .def("NoveltySearchTick", &Population::NoveltySearchTick, py::arg("successful_genome"), "C++ novelty search: one Tick() with novelty as fitness. Returns True if a successful behavior was found.")
            .def("Save", [](Population &p, const std::string &a_filename) { p.Save(a_filename.c_str()); }, py::arg("filename"), "Saves the parameters, the innovation database and all genomes to a text file.")
            .def("GetBestFitnessEver", &Population::GetBestFitnessEver, "The best fitness seen during the whole run.")
            .def("GetBestGenome", &Population::GetBestGenome, "A copy of the best genome of the current population.")
            .def("GetSearchMode", &Population::GetSearchMode, "The phase of the phased search: COMPLEXIFYING, SIMPLIFYING or BLENDED.")
            .def("GetCurrentMPC", &Population::GetCurrentMPC, "The mean population complexity (average number of links).")
            .def("GetBaseMPC", &Population::GetBaseMPC, "The complexity at the start of the current complexifying phase.")
            .def("GetStagnation", &Population::GetStagnation, "Generations since the best fitness last improved.")
            .def("GetMPCStagnation", &Population::GetMPCStagnation, "Generations since the complexity last decreased (simplifying phase).")
            .def("GetGeneration", &Population::GetGeneration, "The current generation number.")
            .def("AccessGenomeByIndex", &Population::AccessGenomeByIndex, py::return_value_policy::reference_internal, py::arg("index"), "A reference to the genome with this index (counting through all species).")
            .def("AccessGenomeByID", &Population::AccessGenomeByID, py::return_value_policy::reference_internal, py::arg("id"), "A reference to the genome with this ID.")
            .def("NumGenomes", &Population::NumGenomes, "The number of genomes in the population.")
            .def_readwrite("Species", &Population::m_Species)
            .def_readwrite("Parameters", &Population::m_Parameters)
            .def_readwrite("RNG", &Population::m_RNG)
            .def_readwrite("ID", &Population::m_ID)
            ;
    population.def(py::pickle(
        [](const Population &a_obj)
        {
            return py::bytes(SerializeToString(a_obj));
        },
        [](const py::bytes &a_state)
        {
            Population *t_obj = new Population();
            DeserializeFromString(static_cast<std::string>(a_state), *t_obj);
            return t_obj;
        }));

///////////////////////////////////////////////////////////////////
// Parameters class
///////////////////////////////////////////////////////////////////

    parameters
            .def(py::init<>())
            .def("Reset", &Parameters::Reset, "Restores all defaults.")
            .def("Load", [](Parameters &p, const std::string &a_filename) { return p.Load(a_filename.c_str()); }, py::arg("filename"),
                 "Loads the parameters from a file. Returns 0 on success.")
            .def("Save", [](Parameters &p, const std::string &a_filename) { p.Save(a_filename.c_str()); }, py::arg("filename"), "Saves the parameters to a text file.")
            .def("ListNeuronTraitParameters", &Parameters::ListNeuronTraitParameters, "The names of the neuron traits.")
            .def("ListLinkTraitParameters", &Parameters::ListLinkTraitParameters, "The names of the link traits.")
            .def("ListGenomeTraitParameters", &Parameters::ListGenomeTraitParameters, "The names of the genome traits.")
            .def("SetNeuronTraitParameters", &Parameters::SetNeuronTraitParameters, py::arg("name"), py::arg("trait_params"), "Defines a neuron trait (see the traits guide).")
            .def("SetLinkTraitParameters", &Parameters::SetLinkTraitParameters, py::arg("name"), py::arg("trait_params"), "Defines a link trait.")
            .def("SetGenomeTraitParameters", &Parameters::SetGenomeTraitParameters, py::arg("name"), py::arg("trait_params"), "Defines a genome trait.")
            .def("GetNeuronTraitParameters", &Parameters::GetNeuronTraitParameters, py::arg("name"))
            .def("GetLinkTraitParameters", &Parameters::GetLinkTraitParameters, py::arg("name"))
            .def("GetGenomeTraitParameters", &Parameters::GetGenomeTraitParameters, py::arg("name"))
            .def("ClearNeuronTraitParameters", &Parameters::ClearNeuronTraitParameters)
            .def("ClearLinkTraitParameters", &Parameters::ClearLinkTraitParameters)
            .def("ClearGenomeTraitParameters", &Parameters::ClearGenomeTraitParameters)

            .def_readwrite("PopulationSize", &Parameters::PopulationSize)
            .def_readwrite("Speciation", &Parameters::Speciation)
            .def_readwrite("DynamicCompatibility", &Parameters::DynamicCompatibility)
            .def_readwrite("MinSpecies", &Parameters::MinSpecies)
            .def_readwrite("MaxSpecies", &Parameters::MaxSpecies)
            .def_readwrite("InnovationsForever", &Parameters::InnovationsForever)
            .def_readwrite("AllowClones", &Parameters::AllowClones)
            .def_readwrite("ArchiveEnforcement", &Parameters::ArchiveEnforcement)
            .def_readwrite("NormalizeGenomeSize", &Parameters::NormalizeGenomeSize)
            .def_property("CustomConstraints", ObjectGetter(&Parameters::pyCustomConstraints), ObjectSetter(&Parameters::pyCustomConstraints))
            .def_property("BehaviorGetter", ObjectGetter(&Parameters::pyBehaviorGetter), ObjectSetter(&Parameters::pyBehaviorGetter))
            .def_readwrite("ConstraintTrials", &Parameters::ConstraintTrials)
            .def_readwrite("YoungAgeTreshold", &Parameters::YoungAgeTreshold)
            .def_readwrite("YoungAgeFitnessBoost", &Parameters::YoungAgeFitnessBoost)
            .def_readwrite("SpeciesMaxStagnation", &Parameters::SpeciesMaxStagnation)
            .def_readwrite("StagnationDelta", &Parameters::StagnationDelta)
            .def_readwrite("OldAgeTreshold", &Parameters::OldAgeTreshold)
            .def_readwrite("OldAgePenalty", &Parameters::OldAgePenalty)
            .def_readwrite("DetectCompetetiveCoevolutionStagnation", &Parameters::DetectCompetetiveCoevolutionStagnation)
            .def_readwrite("KillWorstSpeciesEach", &Parameters::KillWorstSpeciesEach)
            .def_readwrite("KillWorstAge", &Parameters::KillWorstAge)
            .def_readwrite("SurvivalRate", &Parameters::SurvivalRate)
            .def_readwrite("CrossoverRate", &Parameters::CrossoverRate)
            .def_readwrite("OverallMutationRate", &Parameters::OverallMutationRate)
            .def_readwrite("InterspeciesCrossoverRate", &Parameters::InterspeciesCrossoverRate)
            .def_readwrite("MultipointCrossoverRate", &Parameters::MultipointCrossoverRate)
            .def_readwrite("PreferFitterParentRate", &Parameters::PreferFitterParentRate)
            .def_readwrite("RouletteWheelSelection", &Parameters::RouletteWheelSelection)
            .def_readwrite("TournamentSelection", &Parameters::TournamentSelection)
            .def_readwrite("TournamentSize", &Parameters::TournamentSize)
            .def_readwrite("EliteFraction", &Parameters::EliteFraction)
            .def_readwrite("Elitism", &Parameters::EliteFraction) // the name used in parameter files
            .def_readwrite("PhasedSearching", &Parameters::PhasedSearching)
            .def_readwrite("DeltaCoding", &Parameters::DeltaCoding)
            .def_readwrite("SimplifyingPhaseMPCTreshold", &Parameters::SimplifyingPhaseMPCTreshold)
            .def_readwrite("SimplifyingPhaseStagnationTreshold", &Parameters::SimplifyingPhaseStagnationTreshold)
            .def_readwrite("ComplexityFloorGenerations", &Parameters::ComplexityFloorGenerations)
            .def_readwrite("NoveltySearch_K", &Parameters::NoveltySearch_K)
            .def_readwrite("NoveltySearch_P_min", &Parameters::NoveltySearch_P_min)
            .def_readwrite("NoveltySearch_Dynamic_Pmin", &Parameters::NoveltySearch_Dynamic_Pmin)
            .def_readwrite("NoveltySearch_No_Archiving_Stagnation_Treshold", &Parameters::NoveltySearch_No_Archiving_Stagnation_Treshold)
            .def_readwrite("NoveltySearch_Pmin_lowering_multiplier", &Parameters::NoveltySearch_Pmin_lowering_multiplier)
            .def_readwrite("NoveltySearch_Pmin_min", &Parameters::NoveltySearch_Pmin_min)
            .def_readwrite("NoveltySearch_Quick_Archiving_Min_Evaluations", &Parameters::NoveltySearch_Quick_Archiving_Min_Evaluations)
            .def_readwrite("NoveltySearch_Pmin_raising_multiplier", &Parameters::NoveltySearch_Pmin_raising_multiplier)
            .def_readwrite("NoveltySearch_Recompute_Sparseness_Each", &Parameters::NoveltySearch_Recompute_Sparseness_Each)
            .def_readwrite("MutateAddNeuronProb", &Parameters::MutateAddNeuronProb)
            .def_readwrite("SplitRecurrent", &Parameters::SplitRecurrent)
            .def_readwrite("SplitLoopedRecurrent", &Parameters::SplitLoopedRecurrent)
            .def_readwrite("NeuronTries", &Parameters::NeuronTries)
            .def_readwrite("MutateAddLinkProb", &Parameters::MutateAddLinkProb)
            .def_readwrite("MutateAddLinkFromBiasProb", &Parameters::MutateAddLinkFromBiasProb)
            .def_readwrite("MutateRemLinkProb", &Parameters::MutateRemLinkProb)
            .def_readwrite("MutateRemSimpleNeuronProb", &Parameters::MutateRemSimpleNeuronProb)
            .def_readwrite("LinkTries", &Parameters::LinkTries)
            .def_readwrite("MaxLinks", &Parameters::MaxLinks)
            .def_readwrite("MaxNeurons", &Parameters::MaxNeurons)
            .def_readwrite("RecurrentProb", &Parameters::RecurrentProb)
            .def_readwrite("RecurrentLoopProb", &Parameters::RecurrentLoopProb)
            .def_readwrite("MutateWeightsProb", &Parameters::MutateWeightsProb)
            .def_readwrite("MutateWeightsSevereProb", &Parameters::MutateWeightsSevereProb)
            .def_readwrite("WeightMutationRate", &Parameters::WeightMutationRate)
            .def_readwrite("WeightMutationMaxPower", &Parameters::WeightMutationMaxPower)
            .def_readwrite("WeightReplacementRate", &Parameters::WeightReplacementRate)
            .def_readwrite("WeightReplacementMaxPower", &Parameters::WeightReplacementMaxPower)
            .def_readwrite("MaxWeight", &Parameters::MaxWeight)
            .def_readwrite("MinWeight", &Parameters::MinWeight)
            .def_readwrite("MutateActivationAProb", &Parameters::MutateActivationAProb)
            .def_readwrite("MutateActivationBProb", &Parameters::MutateActivationBProb)
            .def_readwrite("ActivationAMutationMaxPower", &Parameters::ActivationAMutationMaxPower)
            .def_readwrite("ActivationBMutationMaxPower", &Parameters::ActivationBMutationMaxPower)
            .def_readwrite("MinActivationA", &Parameters::MinActivationA)
            .def_readwrite("MaxActivationA", &Parameters::MaxActivationA)
            .def_readwrite("MinActivationB", &Parameters::MinActivationB)
            .def_readwrite("MaxActivationB", &Parameters::MaxActivationB)
            .def_readwrite("TimeConstantMutationMaxPower", &Parameters::TimeConstantMutationMaxPower)
            .def_readwrite("BiasMutationMaxPower", &Parameters::BiasMutationMaxPower)
            .def_readwrite("MutateNeuronTimeConstantsProb", &Parameters::MutateNeuronTimeConstantsProb)
            .def_readwrite("MutateNeuronBiasesProb", &Parameters::MutateNeuronBiasesProb)
            .def_readwrite("MinNeuronTimeConstant", &Parameters::MinNeuronTimeConstant)
            .def_readwrite("MaxNeuronTimeConstant", &Parameters::MaxNeuronTimeConstant)
            .def_readwrite("MinNeuronBias", &Parameters::MinNeuronBias)
            .def_readwrite("MaxNeuronBias", &Parameters::MaxNeuronBias)
            .def_readwrite("MutateNeuronActivationTypeProb", &Parameters::MutateNeuronActivationTypeProb)
            .def_readwrite("ActivationFunction_SignedSigmoid_Prob", &Parameters::ActivationFunction_SignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_UnsignedSigmoid_Prob", &Parameters::ActivationFunction_UnsignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_Tanh_Prob", &Parameters::ActivationFunction_Tanh_Prob)
            .def_readwrite("ActivationFunction_TanhCubic_Prob", &Parameters::ActivationFunction_TanhCubic_Prob)
            .def_readwrite("ActivationFunction_SignedStep_Prob", &Parameters::ActivationFunction_SignedStep_Prob)
            .def_readwrite("ActivationFunction_UnsignedStep_Prob", &Parameters::ActivationFunction_UnsignedStep_Prob)
            .def_readwrite("ActivationFunction_SignedGauss_Prob", &Parameters::ActivationFunction_SignedGauss_Prob)
            .def_readwrite("ActivationFunction_UnsignedGauss_Prob", &Parameters::ActivationFunction_UnsignedGauss_Prob)
            .def_readwrite("ActivationFunction_Abs_Prob", &Parameters::ActivationFunction_Abs_Prob)
            .def_readwrite("ActivationFunction_SignedSine_Prob", &Parameters::ActivationFunction_SignedSine_Prob)
            .def_readwrite("ActivationFunction_UnsignedSine_Prob", &Parameters::ActivationFunction_UnsignedSine_Prob)
            .def_readwrite("ActivationFunction_Linear_Prob", &Parameters::ActivationFunction_Linear_Prob)
            .def_readwrite("ActivationFunction_Relu_Prob", &Parameters::ActivationFunction_Relu_Prob)
            .def_readwrite("ActivationFunction_Softplus_Prob", &Parameters::ActivationFunction_Softplus_Prob)
            .def_readwrite("DontUseBiasNeuron", &Parameters::DontUseBiasNeuron)
            .def_readwrite("AllowLoops", &Parameters::AllowLoops)
            .def_readwrite("MutateGenomeTraitsProb", &Parameters::MutateGenomeTraitsProb)
            .def_readwrite("MutateNeuronTraitsProb", &Parameters::MutateNeuronTraitsProb)
            .def_readwrite("MutateLinkTraitsProb", &Parameters::MutateLinkTraitsProb)
            .def_readwrite("DisjointCoeff", &Parameters::DisjointCoeff)
            .def_readwrite("ExcessCoeff", &Parameters::ExcessCoeff)
            .def_readwrite("WeightDiffCoeff", &Parameters::WeightDiffCoeff)
            .def_readwrite("ActivationADiffCoeff", &Parameters::ActivationADiffCoeff)
            .def_readwrite("ActivationBDiffCoeff", &Parameters::ActivationBDiffCoeff)
            .def_readwrite("TimeConstantDiffCoeff", &Parameters::TimeConstantDiffCoeff)
            .def_readwrite("BiasDiffCoeff", &Parameters::BiasDiffCoeff)
            .def_readwrite("ActivationFunctionDiffCoeff", &Parameters::ActivationFunctionDiffCoeff)
            .def_readwrite("CompatTreshold", &Parameters::CompatTreshold)
            .def_readwrite("MinCompatTreshold", &Parameters::MinCompatTreshold)
            .def_readwrite("CompatTresholdModifier", &Parameters::CompatTresholdModifier)
            .def_readwrite("CompatTreshChangeInterval_Generations", &Parameters::CompatTreshChangeInterval_Generations)
            .def_readwrite("CompatTreshChangeInterval_Evaluations", &Parameters::CompatTreshChangeInterval_Evaluations)
            .def_readwrite("MinDeltaCompatEqualGenomes", &Parameters::MinDeltaCompatEqualGenomes)

            .def_readwrite("DivisionThreshold", &Parameters::DivisionThreshold)
            .def_readwrite("VarianceThreshold", &Parameters::VarianceThreshold)
            .def_readwrite("BandThreshold", &Parameters::BandThreshold)
            .def_readwrite("InitialDepth", &Parameters::InitialDepth)
            .def_readwrite("MaxDepth", &Parameters::MaxDepth)
            .def_readwrite("IterationLevel", &Parameters::IterationLevel)
            .def_readwrite("CPPN_Bias", &Parameters::CPPN_Bias)
            .def_readwrite("Width", &Parameters::Width)
            .def_readwrite("Height", &Parameters::Height)
            .def_readwrite("Qtree_Y", &Parameters::Qtree_Y)
            .def_readwrite("Qtree_X", &Parameters::Qtree_X)
            .def_readwrite("Leo", &Parameters::Leo)
            .def_readwrite("LeoThreshold", &Parameters::LeoThreshold)
            .def_readwrite("LeoSeed", &Parameters::LeoSeed)
            .def_readwrite("GeometrySeed", &Parameters::GeometrySeed)
            ;
    AddPickling<Parameters>(parameters);
}

#endif // PYTHON_BINDINGS
