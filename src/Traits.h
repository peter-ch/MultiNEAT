//
// Created by peter on 28.04.17.
//

#ifndef MULTINEAT_TRAITS_H
#define MULTINEAT_TRAITS_H

#include <string>
#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <cmath>

namespace bs = boost;

namespace NEAT
{
    typedef bs::variant<int, bool, double, std::string> TraitType;

    class IntTraitParameters
    {
    public:
        int min, max;
        int mut_power; // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating
    };
    class FloatTraitParameters
    {
    public:
        double min, max;
        double mut_power; // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating
    };
    class StringTraitParameters
    {
    public:
        std::vector<std::string> set; // the set of possible strings
        std::vector<double> probs; // their respective probabilities for appearance
    };

    class TraitParameters
    {
    public:
        double m_ImportanceCoeff;
        double m_MutationProb;

        std::string type; // can be "int", "bool", "float", "string"
        bs::variant<IntTraitParameters, FloatTraitParameters, StringTraitParameters> m_Details;
    };

    class Trait
    {
    public:
        TraitType value;

        Trait()
        {

        }

        Trait(TraitType t)
        {
            value = t;
        }

        Trait(int t)
        {
            value = t;
        }

        Trait(bool t)
        {
            value = t;
        }

        Trait(double t)
        {
            value = t;
        }

        Trait(std::string t)
        {
            value = t;
        }
    };

}
#endif //MULTINEAT_TRAITS_H
