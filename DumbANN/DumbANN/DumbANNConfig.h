#pragma once

#define ENABLE_MICROPROFILE		1

#if	ENABLE_MICROPROFILE
#	include "microprofile.h"
#else
#	define	MICROPROFILE_SCOPEI(_a, _b, _c)
#endif

