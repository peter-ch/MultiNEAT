#ifndef MULTINEAT_NEAT_ASSERT_H
#define MULTINEAT_NEAT_ASSERT_H

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

// Note: this file must not be called Assert.h - on case-insensitive file systems
// it would shadow the standard <assert.h>.

#include <iostream>
#include <sstream>
#include <stdexcept>

// kill any existing declarations
#ifdef ASSERT
#undef ASSERT
#endif

#ifdef VERIFY
#undef VERIFY
#endif

#ifdef DEBUG

//--------------
//  debug macros
//--------------

#define ASSERT(expr)\
        do {\
            if( !(expr) )\
            {\
                std::ostringstream t_assert_msg;\
                t_assert_msg << "ASSERT failed: " << __FILE__ << ", line " << __LINE__ << ": " << #expr;\
                std::cerr << "\n*** " << t_assert_msg.str() << " ***\n\n";\
                throw std::logic_error(t_assert_msg.str());\
            }\
        } while (0)

#define VERIFY(expr)\
        do {\
            if( !(expr) )\
            {\
                std::cerr << "\n*** VERIFY FAILED ***\n" << \
                __FILE__ ", line " << __LINE__ << ": " << \
                #expr << " is false\n\n";\
            }\
        } while (0)

#else // DEBUG

//--------------
//  release macros
//--------------

// ASSERT gets optimised out completely
#define ASSERT(expr) do { } while (0)

// verify has expression evaluated, but no further action taken
#define VERIFY(expr) do { (void)(expr); } while (0)

#endif

#endif // MULTINEAT_NEAT_ASSERT_H
