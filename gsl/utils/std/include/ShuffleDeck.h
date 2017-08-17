// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
