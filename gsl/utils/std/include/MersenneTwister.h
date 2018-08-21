/*
  A C-program for MT19937, with initialization improved 2002/1/26.
  Coded by Takuji Nishimura and Makoto Matsumoto.
 
  Before using, initialize the state by using init_genrand(seed)
  or init_by_array(init_key, key_length).
 
  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
  All rights reserved.
 
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
 
  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
 
  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
 
  3. The names of its contributors may not be used to endorse or promote
  products derived from this software without specific prior written
  permission.
 
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 
  Any feedback is very welcome.
  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
  email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/*************************************************************************
  This pseudo random number generation class was written based on
  mt19937ar.c which was written by M. Matsumoto and T. Nishimura.
  Mt19937ar.c can be used freely for any purpose including commercial 
  distribution. The license information can be found in
  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/elicense.html 

  The Mersenne Twister algorithm is described in the following paper:
  M. Matsumoto and T. Nishimura, "Mersenne Twister: A 623-dimensionally 
  equidistributed uniform pseudorandom number generator", ACM Trans. on 
  Modeling and Computer Simulation Vol. 8, No. 1, January pp.3-30 (1998)
  This so called "Mersenne Twister" alogrithm is proven to be able to
  generate equidistributed random numbers in 623 dimensions with a 
  colossal period of 2^19937 - 1 iterations, though it is not
  cryptographically secure. 
                                               -- Jizhu Lu 02/09/2006
*************************************************************************/

#ifndef MersenneTwister_H
#define MersenneTwister_H

#include <iostream>
#include <math.h>
#include <cassert>

static const char* MT_copyright = "Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, All rights reserved.";

class MersenneTwister {
 private:
  const unsigned _constantN;
  const unsigned _constantM;
  const unsigned _constantU;
  const unsigned _constantS;
  const unsigned _constantT;
  const unsigned _constantL;
  const unsigned long _upperMask;
  const unsigned long _lowerMask;
  const unsigned long _constantB;
  const unsigned long _constantC;

  unsigned long mag01[2];
  unsigned long _stateVector[624];
  unsigned _stateVectorIndex;

 protected:
  void reSeed(unsigned long seed, unsigned rank) {
    seed+=rank;
    mag01[0] = 0x0UL;
    mag01[1] = 0x9908b0dfUL;
    _stateVector[0]= seed & 0xffffffffUL;
    for (_stateVectorIndex=1; _stateVectorIndex<_constantN; _stateVectorIndex++) {
      _stateVector[_stateVectorIndex] = 
        (1812433253UL * (_stateVector[_stateVectorIndex-1] ^ 
                         (_stateVector[_stateVectorIndex-1] >> 30)) + _stateVectorIndex); 
      _stateVector[_stateVectorIndex] &= 0xffffffffUL;
    }
  }

  void reSeedShared(unsigned long seed) {
    mag01[0] = 0x0UL;
    mag01[1] = 0x9908b0dfUL;
    _stateVector[0]= seed & 0xffffffffUL;
    for (_stateVectorIndex=1; _stateVectorIndex<_constantN; _stateVectorIndex++) {
      _stateVector[_stateVectorIndex] = 
        (1812433253UL * (_stateVector[_stateVectorIndex-1] ^ 
                         (_stateVector[_stateVectorIndex-1] >> 30)) + _stateVectorIndex); 
      _stateVector[_stateVectorIndex] &= 0xffffffffUL;
    }
  }

 public:
  MersenneTwister& operator=(const MersenneTwister& rv);

 MersenneTwister(unsigned long seed = 5489UL)
   : _constantN(624), _constantM(397), _constantU(11), _constantS(7),
    _constantT(15), _constantL(18), _upperMask(0x80000000UL), 
    _lowerMask(0x7fffffffUL), _constantB(0x9d2c5680UL),
    _constantC(0xefc60000UL)
      {
        reSeed(seed,0);
      }

 MersenneTwister(unsigned long init_key[], unsigned key_length) 
   : _constantN(624), _constantM(397), _constantU(11), _constantS(7),
    _constantT(15), _constantL(18), _upperMask(0x80000000UL), 
    _lowerMask(0x7fffffffUL), _constantB(0x9d2c5680UL),
    _constantC(0xefc60000UL)
      {
        unsigned i, j, k;
        reSeed(19650218UL,0);
        i=1; j=0;
        k = (_constantN>key_length ? _constantN : key_length);
        for (; k; k--) {
          _stateVector[i] = (_stateVector[i] ^ ((_stateVector[i-1] ^ 
                                                 (_stateVector[i-1] >> 30)) * 1664525UL)) + init_key[j] + j;
          _stateVector[i] &= 0xffffffffUL;
          i++; j++;
          if (i>=_constantN) {
            _stateVector[0] = _stateVector[_constantN-1];
            i=1;
          }
          if (j>=key_length) j=0;
        }
        for (k=_constantN-1; k; k--) {
          _stateVector[i] = (_stateVector[i]^((_stateVector[i-1]^(_stateVector[i-1] >> 30))
                                              * 1566083941UL)) - i;
          _stateVector[i] &= 0xffffffffUL;
          i++;
          if (i>=_constantN) { 
            _stateVector[0] = _stateVector[_constantN-1]; 
            i=1;
          }
        }
        _stateVector[0] = 0x80000000UL;
      }
  
 MersenneTwister(const MersenneTwister& m)
   : _constantN(m._constantN),
    _constantM(m._constantM),
    _constantU(m._constantU),
    _constantS(m._constantS),
    _constantT(m._constantT),
    _constantL(m._constantL),
    _upperMask(m._upperMask),
    _lowerMask(m._lowerMask),
    _constantB(m._constantB),
    _constantC(m._constantC),
    _stateVectorIndex(m._stateVectorIndex)
      {
        for (int i=0; i<2; ++i) {
          mag01[i] = m.mag01[i];
        }
        for (int i=0; i<624; ++i) {
          _stateVector[i] = m._stateVector[i];
        }
      }

  unsigned long irandom32(void)
  {
    unsigned long y;

    if (_stateVectorIndex >= _constantN) {
      unsigned i, j = _constantN-_constantM, k = _constantN-1;

      for (i=0;i<j;i++) {
        y = (_stateVector[i]&_upperMask)|(_stateVector[i+1]&_lowerMask);
        _stateVector[i] = _stateVector[i+_constantM] ^ (y >> 1) ^ mag01[y & 0x1UL];
      }
      for (;i<k;i++) {
        y = (_stateVector[i]&_upperMask)|(_stateVector[i+1]&_lowerMask);
        _stateVector[i] = _stateVector[i-j] ^ (y >> 1) ^ mag01[y & 0x1UL];
      }
      y = (_stateVector[k]&_upperMask)|(_stateVector[0]&_lowerMask);
      _stateVector[k] = _stateVector[_constantM-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

      _stateVectorIndex = 0;
    }
 
    y = _stateVector[_stateVectorIndex++];

    y ^= (y >> _constantU);
    y ^= (y << _constantS) & _constantB;
    y ^= (y << _constantT) & _constantC;
    y ^= (y >> _constantL);

    return y;
  }

  inline double drandom32(void)
  {
    return (double)irandom32()/4294967296.0; 
  }


};

inline MersenneTwister& MersenneTwister::operator=(const MersenneTwister& rv)
{
  if (this == &rv) {
    return *this;
  }
  assert(_constantN==rv._constantN);
  assert(_constantM==rv._constantM);
  assert(_constantU==rv._constantU);
  assert(_constantS==rv._constantS);
  assert(_constantT==rv._constantT);
  assert(_constantL==rv._constantL);
  assert(_upperMask==rv._upperMask);
  assert(_lowerMask==rv._lowerMask);
  assert(_constantB==rv._constantB);
  assert(_constantC==rv._constantC);

  _stateVectorIndex=rv._stateVectorIndex;
  for (int i=0; i<2; ++i) {
    mag01[i] = rv.mag01[i];
  }
  for (int i=0; i<624; ++i) {
    _stateVector[i] = rv._stateVector[i];
  }
  return *this;
}

  class MersenneTwister_S: public MersenneTwister
  {
  public:
    void reSeed(unsigned long seed, unsigned rank) {
      MersenneTwister::reSeed(seed, rank);
    }

    void reSeedShared(unsigned long seed) {
      MersenneTwister::reSeedShared(seed);
    }
  };
#endif
