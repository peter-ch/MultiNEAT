//
// Created by peter on 28.04.17.
//

#ifndef MULTINEAT_TRAITS_H
#define MULTINEAT_TRAITS_H

#include <string>
#include <vector>
#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <cmath>

#ifdef USE_BOOST_PYTHON
#include <boost/python.hpp>
#endif
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>

namespace bs = boost;
#ifdef USE_BOOST_PYTHON
namespace py = bs::python;
#endif

namespace NEAT
{
    class intsetelement
    {
    public:
        // Comparison operator
        bool operator==(const intsetelement& rhs) const
        {
            return rhs.value == value;
        }

        int value;
    };
    class floatsetelement
    {
    public:
        // Comparison operator
        bool operator==(const floatsetelement& rhs) const
        {
            return rhs.value == value;
        }

        double value;
    };
    
    typedef bs::variant<int, double, std::string, intsetelement, floatsetelement
#ifdef USE_BOOST_PYTHON
  , py::object
#endif
    > TraitType;

    class IntTraitParameters
    {
    public:
        int min, max;
        int mut_power; // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating

        IntTraitParameters()
        {
            min = 0; max = 0;
            mut_power = 0;
            mut_replace_prob = 0;
        }
    };
    class FloatTraitParameters
    {
    public:
        double min, max;
        double mut_power; // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating

        FloatTraitParameters()
        {
            min = 0; max = 0;
            mut_power = 0;
            mut_replace_prob = 0;
        }
    };
    class StringTraitParameters
    {
    public:
        std::vector<std::string> set; // the set of possible strings
        std::vector<double> probs; // their respective probabilities for appearance
    };
    class IntSetTraitParameters
    {
    public:
        std::vector<intsetelement> set; // the set of possible ints
        std::vector<double> probs; // their respective probabilities for appearance
    };
    class FloatSetTraitParameters
    {
    public:
        std::vector<floatsetelement> set; // the set of possible floats
        std::vector<double> probs; // their respective probabilities for appearance
    };


    class TraitParameters
    {
    public:
        double m_ImportanceCoeff;
        double m_MutationProb;

        std::string type; // can be "int", "float", "string", "intset", "floatset", "pyobject"
        bs::variant<IntTraitParameters,
                    FloatTraitParameters,
                    StringTraitParameters,
                    IntSetTraitParameters,
                    FloatSetTraitParameters
#ifdef USE_BOOST_PYTHON
                  , py::object
#endif
        > m_Details;

        std::string dep_key; // counts only if this other trait exists..
        std::vector<TraitType> dep_values; // and has one of these values

        // keep dep_key empty and no conditional logic will apply

        TraitParameters()
        {
            m_ImportanceCoeff = 0;
            m_MutationProb = 0;
            type = "int";
            m_Details = IntTraitParameters();
            dep_key = "";
            dep_values.push_back( std::string("") );
        }
    };

    class Trait
    {
    public:
        TraitType value;

        Trait()
        {
            value = 0;
            dep_values.push_back(0);
            dep_key = "";
        }

        std::string dep_key; // counts only if this other trait exists..
        std::vector<TraitType> dep_values; // and has this value
    };

}
#endif //MULTINEAT_TRAITS_H
