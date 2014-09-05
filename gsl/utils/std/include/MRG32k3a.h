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

// *************************************************************
// Revision File : MRG32k3a.h
// Date : March 23, 2012
// Revision Author : JRK
//
// Original Files: MRG32k3aEngine.hpp,MRG32k3aEngine.cpp
// Date: October 4, 2007
// Original Author: RSG
// Classes:  MRG32k3aEngine
// Inheritance: RandomNumberGeneratorIf<MRG32k3aEngine>
// Description: Multiple recursive random number generator based on
//              the description in "Good Parameters and
//              Implementations for Combined Multiple Recursive Random
//              Number Generators" by P. L'Ecuyer 
//              in Section 3 and on the example floating point
//              implementation provided in Figure I.
//              http://www.iro.umontreal.ca/~lecuyer/myftp/papers/combmrg2.ps
//              A shorter version of this paper was published in
//              Operations Research, 47, 1 (1999), 159--164
//
// *************************************************************

#ifndef MRG2K3A_H
#define MRG2K3A_H
#define CM0 4294967087.0
#define CM1 4294944443.0
#define CA02   1403580.0
#define CA03   -810728.0
#define CA11    527612.0
#define CA13  -1370589.0
#define CNORM 2.328306549295727688239738e-10 // 1.0/(CM0+1)
#define CM0INV 2.3283065498378288271152937941012e-10
#define CM1INV 2.3283188252407389754913297722487e-10

#include <stdint.h>
#include <math.h>
#include <cassert>
#include <limits.h>

// rotate variable stored in uint left by number of bits in
// second argument 
inline unsigned rol( const unsigned n, const unsigned long s )
{
  return( ( n << s ) | ( n >> ( 8*sizeof(unsigned) - s ) ) );
}

class MRG32k3a
{
private:
  
  // data

  double mX0_N0;
  double mX0_N1;
  double mX0_N2;
  double mX1_N0;
  double mX1_N1;
  double mX1_N2;

  // methods:

public:

  MRG32k3a()
  {
    mX0_N0 = 12345;
    mX0_N1 = 12345;
    mX0_N2 = 12345;
    mX1_N0 = 12345;
    mX1_N1 = 12345;
    mX1_N2 = 12345;
  }

  MRG32k3a(const class MRG32k3a& rv)
  {
    mX0_N0 = rv.mX0_N0;
    mX0_N1 = rv.mX0_N1;
    mX0_N2 = rv.mX0_N2;
    mX1_N0 = rv.mX1_N0;
    mX1_N1 = rv.mX1_N1;
    mX1_N2 = rv.mX1_N2;
  }

  MRG32k3a& operator=(const MRG32k3a& rv);

  void reSeed(unsigned seed, unsigned rank) {
    seed+=rank;
    reSeedShared(seed);
  }

  void reSeedShared(unsigned seed) {
    // A better way to handle this would be to use the seed as the
    // index of the substream to be used--that would guarantee no 
    // overlap between between engines initialized with different
    // seeds (at least for less than 2**76 iterations)
    mX0_N0 = 12345;
    mX0_N1 = 12345;
    mX0_N2 = 12345;
    mX1_N0 = 12345;
    mX1_N1 = 12345;
    mX1_N2 = 12345;
    
    if ( seed == 0 ) return;
    
    unsigned mask = rol( seed, 5 );
    mX0_N0 = static_cast<const double>(static_cast<const uint32_t>(mX0_N0) ^ mask);
    mask = rol( seed, 10 );
    mX0_N1 = static_cast<const double>(static_cast<const uint32_t>(mX0_N1) ^ mask);
    mask = rol( seed, 15 );
    mX0_N2 = static_cast<const double>(static_cast<const uint32_t>(mX0_N2) ^ mask);
    mask = rol( seed, 20 );
    mX1_N0 = static_cast<const double>(static_cast<const uint32_t>(mX0_N0) ^ mask);
    mask = rol( seed, 25 );
    mX1_N1 = static_cast<const double>(static_cast<const uint32_t>(mX0_N1) ^ mask);
    mask = rol( seed, 30 );
    mX1_N2 = static_cast<const double>(static_cast<const uint32_t>(mX0_N2) ^ mask);
  }

  double drandom32(void)
  {
    // First component
    double s0 = CA02*mX0_N1 + CA03*mX0_N0;
    int32_t i = static_cast<int32_t>(s0*CM0INV);
    s0 -= i*CM0;
    s0 = s0 < 0 ? s0 + CM0 : s0;
    mX0_N0 = mX0_N1;
    mX0_N1 = mX0_N2;
    mX0_N2 = s0;
    
    // Second component
    double s1 = CA11*mX1_N2 + CA13*mX1_N0;
    i = static_cast<int32_t>(s1*CM1INV);
    s1 -= i*CM1;
    s1 = s1 < 0 ? s1 + CM1 : s1;
    mX1_N0 = mX1_N1;
    mX1_N1 = mX1_N2;
    mX1_N2 = s1;
    
    // Combination of components
    return( s0 <= s1 ? 
	    ( s0 - s1 + CM0 )*CNORM :
	    ( s0 - s1 )*CNORM );
  }
  inline unsigned long irandom32(void)
  {
    return long(floor( LONG_MAX * drandom32() ) );
  }
};

inline MRG32k3a& MRG32k3a::operator=(const MRG32k3a& rv)
{
  if (this == &rv) {
    return *this;
  }

  mX0_N0 = rv.mX0_N0;
  mX0_N1 = rv.mX0_N1;
  mX0_N2 = rv.mX0_N2;
  mX1_N0 = rv.mX1_N0;
  mX1_N1 = rv.mX1_N1;
  mX1_N2 = rv.mX1_N2;

  return *this;
}

#endif
