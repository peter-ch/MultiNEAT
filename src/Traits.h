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
    /*class StringTraitParameters
    {
    public:
    };
    
    class FloatTraitParameters
    {
    public:
        double min, max;
        double mutation_rate;
        double mutation_power;

        // Initialize completely new instance from scratch
        void Init(TraitType& a)
        {
            if (a.type() == typeid(double))
            {
                a = 0.0;
            }
        }
        // Initialize from one or two other instances
        void InitFrom(TraitType& a, const TraitType& other)
        { }
        void InitFrom(TraitType& a, const TraitType& t_1, const TraitType& t_2)
        { }
        // Randomize completely
        void Randomize()
        { }

        void Mutate()
        { }

        double Distance(const TraitType& a_1, const TraitType& a_2)
        {
            return 0.0;
        }

    };
    
    class IntTraitParameters
    {
    public:
    };*/
    
    class TraitParameters
    {
    public:

    };

    class Trait
    {
    public:

    };



}
#endif //MULTINEAT_TRAITS_H
