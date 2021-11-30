
#pragma comment(lib, "Ws2_32.lib")

#include "DumbANN/NeuralNetwork.h"
#include "DumbANN/LayerDense.h"
#include "DumbANN/LayerConv2D.h"

#include <stdlib.h>
#include <time.h>

struct	SMnistLabelHeader
{
	int32_t		m_Magic;
	int32_t		m_LabelCount;
};

struct	SMnistImageHeader
{
	int32_t		m_Magic;
	int32_t		m_ImageCount;
	int32_t		m_Rows;
	int32_t		m_Columns;
};

int32_t	reverseBytes(int i)
{
	uint8_t c1, c2, c3, c4;
	c1 = i & 0xFF;
	c2 = (i >> 8) & 0xFF;
	c3 = (i >> 16) & 0xFF;
	c4 = (i >> 24) & 0xFF;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void	LoadDataSet(std::vector<float> &images,
					std::vector<uint8_t> &labels,
					const char *imageFilePath,
					const char *labelFilePath)
{
	SMnistImageHeader	imgHeader = {};
	SMnistLabelHeader	lblHeader = {};
	FILE				*imageFile = nullptr;
	FILE				*labelFile = nullptr;

	if (fopen_s(&imageFile, imageFilePath, "rb") != 0)
	{
		fprintf(stderr, "Could not open image file");
		return;
	}
	if (fopen_s(&labelFile, labelFilePath, "rb") != 0)
	{
		fprintf(stderr, "Could not open label file");
		return;
	}
	if (imageFile == nullptr || labelFile == nullptr)
	{
		fprintf(stderr, "Could not open file");
		return;
	}
	if (fread(&imgHeader, 1, sizeof(SMnistImageHeader), imageFile) != sizeof(SMnistImageHeader))
	{
		fprintf(stderr, "Could not read image header");
		return;
	}
	if (fread(&lblHeader, 1, sizeof(SMnistLabelHeader), labelFile) != sizeof(SMnistLabelHeader))
	{
		fprintf(stderr, "Could not read label header");
		return;
	}

	// Reverse bytes for headers:
	imgHeader.m_Magic = reverseBytes(imgHeader.m_Magic);
	imgHeader.m_ImageCount = reverseBytes(imgHeader.m_ImageCount);
	imgHeader.m_Rows = reverseBytes(imgHeader.m_Rows);
	imgHeader.m_Columns = reverseBytes(imgHeader.m_Columns);

	lblHeader.m_Magic = reverseBytes(lblHeader.m_Magic);
	lblHeader.m_LabelCount = reverseBytes(lblHeader.m_LabelCount);

	if (imgHeader.m_Magic != 0x00000803 || lblHeader.m_Magic != 0x00000801)
	{
		fprintf(stderr, "Wrong magic number");
		return;
	}

	int						imgReadSize = imgHeader.m_Rows * imgHeader.m_Columns * imgHeader.m_ImageCount;
	std::vector<uint8_t>	labelData;
	std::vector<uint8_t>	imageData;

	if (imgReadSize <= 0 || lblHeader.m_LabelCount <= 0)
	{
		fprintf(stderr, "No data to read from file");
		return;
	}

	labelData.resize(lblHeader.m_LabelCount);
	imageData.resize(imgReadSize);

	if (fread(imageData.data(), 1, imageData.size(), imageFile) != imageData.size())
	{
		fprintf(stderr, "Could not read image data");
		return;
	}
	if (fread(labelData.data(), 1, labelData.size(), labelFile) != labelData.size())
	{
		fprintf(stderr, "Could not read label data");
		return;
	}

	images.resize(imageData.size());
	labels.resize(labelData.size());

	for (size_t i = 0; i < imageData.size(); ++i)
		images[i] = static_cast<float>(imageData[i]) / 255.0f;
	labels = labelData;
}

void	TrainNetwork(CNeuralNetwork &ann, const std::vector<float> &images, const std::vector<uint8_t> &labels)
{
	const int			inputSize = images.size() / labels.size();
	const size_t		epochCount = 100;
	const size_t		batchCount = 10;
	const size_t		miniBatchCount = 5;
	std::vector<float>	expectedOutput(10);

	for (size_t epoch = 0; epoch < epochCount; ++epoch)
	{
		float	errorEpoch = 0.0f;
		float	error = 0.0f;

		for (size_t batchIdx = 0; batchIdx < batchCount; ++batchIdx)
		{
			for (size_t miniBatchIdx = 0; miniBatchIdx < miniBatchCount; ++miniBatchIdx)
			{
				int					randImgIdx = rand() % labels.size();
				uint8_t				curLabel = labels[randImgIdx];
				// Label to output:
				for (size_t j = 0; j < 10; ++j)
					expectedOutput[j] = 0.0f;
				expectedOutput[curLabel] = 1.0f;
				// Input image data:
				const float* inputPtr = images.data() + (ptrdiff_t)randImgIdx * inputSize;
				// Feedforward:
				ann.FeedForward(inputPtr);
				// Compute error:
				float	currentError = 0.0f;
				for (size_t j = 0; j < 10; ++j)
					currentError += abs(expectedOutput[j] - ann.GetOutput().Data()[j]);
				error += currentError;
				errorEpoch += currentError;
				// Backpropagation:
				ann.BackPropagateError(inputPtr, expectedOutput);
			}
			ann.UpdateWeightAndBiases();
		};
		printf("Error for epoch %u/%u is:\t%f\n", (int)epoch, (int)epochCount, errorEpoch / (float)(batchCount * miniBatchCount));
		errorEpoch = 0.0f;
	}
	printf("End of training\n");
}

float	TestNetwork(CNeuralNetwork &ann, const std::vector<float> &images, const std::vector<uint8_t> &labels)
{
	const int			inputSize = images.size() / labels.size();
	std::vector<float>	expectedOutput(10);
	float				error = 0.0f;
	int					errorCount = 0;

	for (size_t i = 0; i < labels.size(); ++i)
	{
		size_t	curLabel = labels[i];
		// Label to output:
		for (size_t j = 0; j < 10; ++j)
			expectedOutput[j] = 0.0f;
		expectedOutput[curLabel] = 1.0f;
		// Input image data:
		const float* inputPtr = images.data() + i * inputSize;
		// Feedforward:
		ann.FeedForward(inputPtr);

		// Compute error:
		float	currentError = 0.0f;
		for (size_t j = 0; j < 10; ++j)
			currentError += abs(expectedOutput[j] - ann.GetOutput().Data()[j]);
		error += currentError;

		// Compute Largest:
		float	largest = -1.0f;
		int		largestIdx = -1;
		float	secondLargest = -1.0f;
		int		secondLargestIdx = -1;
		for (size_t j = 0; j < 10; ++j)
		{
			if (ann.GetOutput().Data()[j] > largest)
			{
				if (largest > 0)
				{
					secondLargest = largest;
					secondLargestIdx = largestIdx;
				}
				largest = ann.GetOutput().Data()[j];
				largestIdx = j;
			}
			else if (ann.GetOutput().Data()[j] > secondLargest)
			{
				secondLargest = ann.GetOutput().Data()[j];
				secondLargestIdx = j;
			}
		}
		if (largestIdx != curLabel)
		{
			++errorCount;
			// printf("\rError: image %lu label is %u guess is %u (%f) and %u (%f)\n", (int)(i + 1), (int)curLabel, largestIdx, largest, secondLargestIdx, secondLargest);
		}

		if ((i + 1) % 100 == 0)
		{
			printf("Testing %lu/%lu error: %f, error count: %u\r", (int)(i + 1), (int)labels.size(), error / 100.0f, errorCount);
			error = 0.0f;
		}
	};
	float	finalError = (float)errorCount / (float)labels.size();
	printf("\nFinal error count: %u/%u (%f%%)\n", (int)errorCount, (int)labels.size(), (1.0f - finalError) * 100.0);
	return finalError;
}

float	TestMNIST()
{
	srand(42);

	printf("--------------------------------\n");
	printf("MNIST Test\n");
	printf("4 Layers:\n");
	printf("\tCLayerDense 28x28->28x28\n");
	printf("\tCLayerDense 28x28->128\n");
	printf("\tCLayerDense 128->64\n");
	printf("\tCLayerDense 64->10\n");

	std::vector<float>		images;
	std::vector<uint8_t>	labels;

	LoadDataSet(images, labels, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");

	if (images.empty() || labels.empty())
		return -1.0f;

	const size_t	inputSize = images.size() / labels.size();
	const size_t	outputSize = 10;
	CNeuralNetwork	ann;
	CLayerDense		layers[4];

	// Output layer is linear to better backpropagate error during training:
	layers[0].Setup(inputSize, inputSize, EActivation::Relu, ERandInitializer::RandHe, EOptimization::SGD);
	layers[1].Setup(inputSize, 128, EActivation::Relu, ERandInitializer::RandHe, EOptimization::SGD);
	layers[2].Setup(128, 64, EActivation::Relu, ERandInitializer::RandHe, EOptimization::SGD);
	layers[3].Setup(64, outputSize, EActivation::Relu, ERandInitializer::RandHe, EOptimization::SGD);

	ann.AddLayer(&layers[0]);
	ann.AddLayer(&layers[1]);
	ann.AddLayer(&layers[2]);
	ann.AddLayer(&layers[3]);

	TrainNetwork(ann, images, labels);

	images.clear();
	labels.clear();

	LoadDataSet(images, labels, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

	float	error = TestNetwork(ann, images, labels);
	printf("--------------------------------\n");
	return error;
}

float	TestXOR()
{
	srand(753);

	printf("--------------------------------\n");
	printf("XOR Test\n");
	printf("2 Layers:\n");
	printf("\tCLayerDense 2->2\n");
	printf("\tCLayerDense 2->1\n");

	CLayerDense		layers[2];

	layers[0].Setup(2, 2);
	layers[1].Setup(2, 1);

	CNeuralNetwork	ann;

	ann.AddLayer(&layers[0]);
	ann.AddLayer(&layers[1]);

	const size_t		trainingCount = 50000;
	float				input[2];
	std::vector<float>	expectedOutput;

	expectedOutput.resize(1);
	for (size_t i = 0; i < trainingCount; ++i)
	{
		int		rand0 = rand() % 2;
		int		rand1 = rand() % 2;
		input[0] = rand0 == 0 ? 0.0f : 1.0f;
		input[1] = rand1 == 0 ? 0.0f : 1.0f;
		expectedOutput[0] = (rand0 ^ rand1) == 0 ? 0.0f : 1.0f;
		ann.FeedForward(input);
		ann.BackPropagateError(input, expectedOutput);
		ann.UpdateWeightAndBiases();
		const float		curError = abs(ann.GetOutput().Data()[0] - expectedOutput[0]);
		printf("training %u/%u error is %f\r", (int)i + 1, (int)trainingCount, curError);
	}

	const size_t		testCount = 100;
	float				error = 0.0f;

	for (size_t i = 0; i < testCount; ++i)
	{
		int		rand0 = rand() % 2;
		int		rand1 = rand() % 2;
		input[0] = rand0 == 0 ? 0.0f : 1.0f;
		input[1] = rand1 == 0 ? 0.0f : 1.0f;
		expectedOutput[0] = (rand0 ^ rand1) == 0 ? 0.0f : 1.0f;
		ann.FeedForward(input);
		error += abs(ann.GetOutput().Data()[0] - expectedOutput[0]);
	}
	const float		avgError = error / testCount;
	printf("\nXOR test error is %f\n", avgError);
	printf("--------------------------------\n");
	return avgError;
}

float	TestCosine()
{
	srand(545);

	printf("--------------------------------\n");
	printf("Cosine Test\n");
	printf("4 Layers:\n");
	printf("\tCLayerDense 1->8\n");
	printf("\tCLayerDense 8->32\n");
	printf("\tCLayerDense 32->16\n");
	printf("\tCLayerDense 16->1\n");

	CLayerDense		layers[4];

	layers[0].Setup(1, 16, EActivation::Sigmoid);
	layers[1].Setup(16, 32, EActivation::Sigmoid);
	layers[2].Setup(32, 16, EActivation::Sigmoid);
	layers[3].Setup(16, 1, EActivation::Linear);

	CNeuralNetwork	ann;

	ann.AddLayer(&layers[0]);
	ann.AddLayer(&layers[1]);
	ann.AddLayer(&layers[2]);
	ann.AddLayer(&layers[3]);

	const size_t		trainingCount = 1000000;
	std::vector<float>	expectedOutput;

	expectedOutput.resize(1);
	for (size_t i = 0; i < trainingCount; ++i)
	{
		float	randX = (float)rand() / (float)RAND_MAX * 3.1415 * 5;
		expectedOutput[0] = cos(randX);
		ann.FeedForward(&randX);
		ann.BackPropagateError(&randX, expectedOutput);
		ann.UpdateWeightAndBiases();
		const float		curError = abs(ann.GetOutput().Data()[0] - expectedOutput[0]);
		if ((i + 1) % 100 == 0)
		printf("training %u/%u error is %f\r", (int)i + 1, (int)trainingCount, curError);
	}

	const size_t		testCount = 100;
	float				testError = 0.0f;

	for (size_t i = 0; i < testCount; ++i)
	{
		float	randX = (float)rand() / (float)RAND_MAX * 3.1415 * 5;
		float	cosRandX = cos(randX);
		ann.FeedForward(&randX);
		testError += abs(ann.GetOutput().Data()[0] - cosRandX);
	}
	printf("\nCosine test error is %f\n", testError / (float)testCount);
	printf("--------------------------------\n");
	return 0.0f;
}