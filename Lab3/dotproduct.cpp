/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu2.hpp>

/* SkePU user functions */

float mult(float x1, float x2)
{
    return x1*x2;
}

float plus(float x1, float x2)
{
    return x1 + x2;
}


int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
//	spec.setCPUThreads(<integer value>);


	/* Skeleton instances */
	auto instance = skepu2::MapReduce<2>(mult, plus);
	auto mapInstance = skepu2::Map<2>(mult);
	auto reduceInstance = skepu2::Reduce(plus);
// ...

	/* Set backend (important, do for all instances!) */
//	instance.setBackend(spec);

	/* SkePU containers */
	skepu2::Vector<float> v1(size, 1.0f), v2(size, 2.0f);


	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu2::benchmark::measureExecTime([&]
	{
        resComb = instance(v1, v2);
	});
	
	auto timeSep = skepu2::benchmark::measureExecTime([&]
	{
		// your code here
        skepu2::Vector<float> mapResult(size);
        mapInstance(mapResult, v1, v2);
        resSep = reduceInstance(mapResult);
	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}

