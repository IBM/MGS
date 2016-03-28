#include "../common/include/NumberUtils.h"
#include <iostream>
#include <gmp.h>

//cd ../common/;make realclean;make; cd -; g++ main_math.cpp -lgmpxx -lgmp -I/packages/gmp/6.0.0/include -g  
int main()
{
	int numChan=4;
	int numChanStates=3;
	int balls = numChan;
	int bins = numChanStates;


	mpz_t fact;
	mpz_init(fact);
	unsigned long N=4;

	//scanf("%lu", &N);
	factorial(fact, N);

	std::cout << "Factorial of "<< N<<"! is "  ;
	std::flush(std::cout);
	char* buf;
	buf = (char*)malloc(mpz_sizeinbase(fact, 10) + 1);
	assert(buf);
	mpz_get_str(buf, 10, fact);
	printf("%s\n", buf);

	free(buf);
	mpz_clear(fact);

	std::cout << "There are " << combination(balls,bins) << " ways to put " << balls << " balls in " << bins << " bins" << std::endl;

	int* StateVector;
	int rows, cols;
	count_ball2bin(balls, bins, StateVector, rows, cols);

	for (int ii = 0; ii< rows; ii++)
	{
		for (int jj=0; jj< cols; jj++)
			std::cout << StateVector[Map1Dindex(ii,jj,cols)] << ",";
		std::cout << std::endl;
	}
	delete StateVector;

}
