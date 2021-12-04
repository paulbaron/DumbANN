
#pragma comment(lib, "Ws2_32.lib")

#include "UnitTests.h"
#include "DumbANN/DumbANNConfig.h"


#include <cstdlib>

int		main(int ac, char** av)
{
#if		ENABLE_MICROPROFILE
	MicroProfileOnThreadCreate("Main");
	MicroProfileSetEnableAllGroups(true);
	MicroProfileSetForceMetaCounters(true);
#endif
//	float	convTest = TestConvolution(false);
//	if (convTest < 0.0f)
//		return EXIT_FAILURE;
//	float	poolTest = TestConvolution(true);
//	if (poolTest < 0.0f)
//		return EXIT_FAILURE;
	float	mnistTest = TestMNIST();
	if (mnistTest < 0.0f)
		return EXIT_FAILURE;
//	float	xorTest = TestXOR();
//	if (xorTest < 0.0f)
//		return EXIT_FAILURE;
//	float	cosTest = TestCosine();
//	if (cosTest < 0.0f)
//		return EXIT_FAILURE;

#if		ENABLE_MICROPROFILE
	MicroProfileDumpFileImmediately("ANN.html", nullptr, nullptr);
	MicroProfileShutdown();
#endif
	return EXIT_SUCCESS;
}