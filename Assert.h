#ifndef INCLUDE_GUARD_Assert_h
#define INCLUDE_GUARD_Assert_h

#include <iostream>
#include <assert.h>

// kill any existing declarations
#ifdef ASSERT
#undef ASSERT
#endif

#ifdef VERIFY
#undef VERIFY
#endif

#ifdef _DEBUG

#ifdef WIN32
//--------------
//  debug macros
//--------------
#define BREAK_CPU()			__asm { int 3 }

#define ASSERT(expr)\
		{\
			if( !(expr) )\
			{\
				std::cout << "\n*** ASSERT1 ***\n" << \
				__FILE__ ", line " << __LINE__ << ": " << \
				#expr << " is false\n\n";\
				BREAK_CPU();\
			}\
		}

#define VERIFY(expr)\
		{\
			if( !(expr) )\
			{\
				std::cout << "\n*** VERIFY FAILED ***\n" << \
				__FILE__ ", line " << __LINE__ << ": " << \
				#expr << " is false\n\n";\
				BREAK_CPU();\
			}\
		}
#else
#define ASSERT(expr)\
		{\
			if( !(expr) )\
			{\
				std::cout << "\n*** ASSERT ***\n"; \
				assert(expr);\
			}\
		}


#define VERIFY(expr)\
		{\
			if( !(expr) )\
			{\
				std::cout << "\n*** VERIFY FAILED ***\n"; \
				assert(expr);\
			}\
		}

#endif
#else // _DEBUG

//--------------
//  release macros
//--------------

// ASSERT gets optimised out completely
#define ASSERT(expr)

// verify has expression evalutated, but no further action taken
#define VERIFY(expr) if( expr ) {}

#endif

#endif // INCLUDE_GUARD_Assert_h
