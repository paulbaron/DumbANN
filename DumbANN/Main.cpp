
#pragma comment(lib, "Ws2_32.lib")

#include "UnitTests.h"

#include <cstdlib>

int		main(int ac, char** av)
{
	float	mnistTest = TestMNIST();
	if (mnistTest < 0.0f)
		return EXIT_FAILURE;
	float	xorTest = TestXOR();
	if (xorTest < 0.0f)
		return EXIT_FAILURE;
	float	cosTest = TestCosine();
	if (cosTest < 0.0f)
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}