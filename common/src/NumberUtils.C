// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "NumberUtils.h"

#include <gmp.h>
#include <assert.h>

//template <typename T>
//T Square(T x)
//{
//  return x * x;
//}

// GOAL:  find factorial
//  result = N!
void factorial(mpz_t result, unsigned long N)
{
  mpz_set_ui(result, 1);
  while (N > 1)
  {
    mpz_mul_ui(result, result, N--);
  }
}

double factorial(int c)
{
    double result = 1;

    for (int n = 1; n <= c; n++)
    {
        result = result * n;
    }

    return result;
}

// GOAL: find (N+M-1)! / (M! * (N-1)!)
// How many ways to put M balls into N bins
// without discriminating balls
unsigned int combination(int M, int N)
{
  mpz_t fact;
  mpz_init(fact);
  unsigned long input = N + M - 1;
  factorial(fact, input);

  mpz_t fact1;
  {
    mpz_init(fact1);
    unsigned long input = N - 1;
    factorial(fact1, input);
  }

  mpz_t fact2;
  {
    mpz_init(fact2);
    unsigned long input = M;
    factorial(fact2, input);
  }
  mpz_t tmp;
  mpz_init(tmp);
  mpz_mul(tmp, fact1, fact2);
  mpz_t result;
  mpz_init(result);
  mpz_cdiv_q(result, fact, tmp);

  unsigned int rval = (unsigned int)mpz_get_ui(result);
  return rval;
}

// NOTE:
//   total bins = 'bins'
//   ball-available = 'balls'
//   bins-available-count-from-ride-side = 'N'
template <typename T>
void combinator_mem_recursive(const int &bins, int balls, int N, int *&matrix,
                              T &count)
{
  for (int ii = balls; ii >= 0; ii--)
  {
    if (N == 1)
    {
      matrix[Map1Dindex(count, bins - N, bins)] = ii;
      count = count + 1;
			return;
    }
    else
    {
      if (balls == 0 and N > 1)
      {
        for (int k = bins - N; k < bins; k++)
          matrix[Map1Dindex(count, k, bins)] = 0;
        count = count + 1;
      }
      else
      {
        matrix[Map1Dindex(count, bins - N, bins)] = ii;
        combinator_mem_recursive(bins, balls - ii, N - 1, matrix, count);
				if (ii > 0 and count > 1)
				{
					for (int jj=0; jj< bins-N; jj++)
						matrix[Map1Dindex(count,jj, bins)] =
							matrix[Map1Dindex(count-1, jj, bins)];
				}
      }
    }
  }
}

// GOAL: find out all configurations for
// putting #balls into #bins
// and store the configurations in 'matrix' matrix
//  row = #configuraitons
//  col = #bins
void count_ball2bin(int balls, int bins, int *&matrix, int &row, int &col)
{
  row = combination(balls, bins);
  col = bins;
  matrix = new int[row * col]();
  int count = 0;  // starting row index
  int N = bins;
  combinator_mem_recursive(bins, balls, N, matrix, count);
  assert(count == row);
}

//GOAL: return the # of non-zero elements
//    in a vector within a given range of indices
//template <typename T>
//unsigned int count_nonzero(T* vector, int offset, int range)
//{
//	unsigned int count= 0;
//	for (int ii=0; ii < range; ii++)
//	{
//    if(vector[offset+ii] > 0) count++;
//	}
//	return count;
//}
