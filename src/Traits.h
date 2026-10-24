//
// Created by peter on 28.04.17.
//

#ifndef MULTINEAT_TRAITS_H
#define MULTINEAT_TRAITS_H

#include <string>
#include <vector>
#include <variant>
#include <cmath>

#ifdef PYTHON_BINDINGS
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace NEAT
{
    class intsetelement
    {
    public:
        int value;

        // Comparison operator
        bool operator==(const intsetelement& rhs) const
        {
            return rhs.value == value;
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & value;
        }
    };
    class floatsetelement
    {
    public:
        
        double value;
        
        // Comparison operator
        bool operator==(const floatsetelement& rhs) const
        {
            return rhs.value == value;
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & value;
        }
    };

    typedef std::variant<int, double, std::string, intsetelement, floatsetelement
#ifdef PYTHON_BINDINGS
  , py::object
#endif
    > TraitType;

    // Compares two trait values. Python objects are compared with Python's == operator.
    inline bool TraitValuesEqual(const TraitType &a, const TraitType &b)
    {
        if (a.index() != b.index())
        {
            return false;
        }
#ifdef PYTHON_BINDINGS
        if (std::holds_alternative<py::object>(a))
        {
            const py::object &x = std::get<py::object>(a);
            const py::object &y = std::get<py::object>(b);
            if (!x || !y)
            {
                return x.ptr() == y.ptr();
            }
            return x.equal(y);
        }
#endif
        if (std::holds_alternative<int>(a)) return std::get<int>(a) == std::get<int>(b);
        if (std::holds_alternative<double>(a)) return std::get<double>(a) == std::get<double>(b);
        if (std::holds_alternative<std::string>(a)) return std::get<std::string>(a) == std::get<std::string>(b);
        if (std::holds_alternative<intsetelement>(a)) return std::get<intsetelement>(a) == std::get<intsetelement>(b);
        if (std::holds_alternative<floatsetelement>(a)) return std::get<floatsetelement>(a) == std::get<floatsetelement>(b);
        return false;
    }

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

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & min;
            ar & max;
            ar & mut_power;
            ar & mut_replace_prob;
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

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & min;
            ar & max;
            ar & mut_power;
            ar & mut_replace_prob;
        }
    };
    class StringTraitParameters
    {
    public:
        std::vector<std::string> set; // the set of possible strings
        std::vector<double> probs; // their respective probabilities for appearance

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & set;
            ar & probs;
        }
    };
    class IntSetTraitParameters
    {
    public:
        std::vector<intsetelement> set; // the set of possible ints
        std::vector<double> probs; // their respective probabilities for appearance

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & set;
            ar & probs;
        }
    };
    class FloatSetTraitParameters
    {
    public:
        std::vector<floatsetelement> set; // the set of possible floats
        std::vector<double> probs; // their respective probabilities for appearance

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & set;
            ar & probs;
        }
    };

    class TraitParameters
    {
    public:
        double m_ImportanceCoeff;
        double m_MutationProb;

        std::string type; // can be "int", "float", "string", "intset", "floatset", "pyobject"
        std::variant<IntTraitParameters,
                    FloatTraitParameters,
                    StringTraitParameters,
                    IntSetTraitParameters,
                    FloatSetTraitParameters
#ifdef PYTHON_BINDINGS
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
            dep_values.emplace_back( std::string("") );
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & m_ImportanceCoeff;
            ar & m_MutationProb;
            ar & type;
            ar & m_Details;
            ar & dep_key;
            ar & dep_values;
        }
    };

    class Trait
    {
    public:
        TraitType value;

        Trait()
        {
            value = 0;
            dep_values.emplace_back(0);
            dep_key = "";
        }

        std::string dep_key; // counts only if this other trait exists..
        std::vector<TraitType> dep_values; // and has this value

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /*version*/)
        {
            ar & value;
            ar & dep_key;
            ar & dep_values;
        }
    };

}
#endif //MULTINEAT_TRAITS_H
