// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_rand_H
#define C_rand_H

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <limits.h>

class C_rand {
   public:
      void reSeed(unsigned int seed, unsigned rank) {
	seed+=rank;
	srand(seed);
      }

      void reSeedShared(unsigned int seed) {
	srand(seed);
      }

      C_rand() {}


      inline double drandom32(void)
      {
	return (double)rand()/_randMaxPlusOne;
      }

      inline unsigned long irandom32(void)
      {
	return long(floor( LONG_MAX * drandom32() ) );
      }
  private:
      static double _randMaxPlusOne;
};

#endif
