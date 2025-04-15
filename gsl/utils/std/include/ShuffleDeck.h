// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SHUFFLEDECK_H
#define SHUFFLEDECK_H
#include "Copyright.h"

#include "rndm.h"
#include <vector>

class ShuffleDeck
{
public:
  ShuffleDeck(int, RNG&);
   ShuffleDeck(ShuffleDeck const *);   
   virtual ~ShuffleDeck();
   int getNext(RNG&);
private:
   void shuffle(RNG&);
   int* _data;
   int _current;
   int _size;
};

#endif
