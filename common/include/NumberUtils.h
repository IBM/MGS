#ifndef _NUMBERUTILS_H_
#define _NUMBERUTILS_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <iostream>
#include <ostream>

//x = row, y=col
//WIDTH=#col, HEIGHT=#row
#ifndef Map1Dindex
#define Map1Dindex(x, y, WIDTH) ((y) + (x) * (WIDTH))
#endif

template <typename T>
T Square(T x)
{
  return x * x;
}

// GOAL:  find factorial
//  result = N!
void factorial(mpz_t result, unsigned long N);

// GOAL: find (N+M-1)! / (M! * (N-1)!)
// How many ways to put M balls into N bins
// without discriminating balls
unsigned int combination(int M, int N);

// NOTE:
//   total bins = 'bins'
//   ball-available = 'balls'
//   bins-available-count-from-ride-side = 'N'
void combinator_mem_recursive(const int &bins, int balls, int N, int *&matrix,
                              unsigned int &count);

// GOAL: find out all configurations for
// putting #balls into #bins
// and store the configurations in 'matrix' matrix
//  row = #configuraitons
//  col = #bins
void count_ball2bin(int balls, int bins, int *&matrix, int &row, int &col);

//GOAL: return the # of non-zero elements
//    in a vector within a given range of indices
template <typename T>
unsigned int count_nonzero(T* vector, int offset, int range)
{
	unsigned int count= 0;
	for (int ii=0; ii < range; ii++)
	{
    if(vector[offset+ii] > 0) count++;
	}
	return count;
}

//GOAL: 
template<typename T>
T linear_interp( T x0, T y0, T x1, T y1, T x )
{
  T a = (y1-y0)/(x1-x0);//tan(alpha)
  T b = -a*x0+y0;
  T y = a * x + b;
  return y;
}
#endif
