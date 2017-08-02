// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ShuffleDeck.h"

ShuffleDeck::ShuffleDeck(int size, RNG& rng)
   : _current(0), _size(size)
{
   _data = new int[_size];
   for (int i=0; i<_size; i++) {
      _data[i]= i;
   }     
   shuffle(rng);
}

ShuffleDeck::ShuffleDeck(ShuffleDeck const *rval)  
   : _data(0), _current(rval->_current), _size(rval->_size)
//    ,_random(rval->_random)
{
   _data = new int[_size];
   // May use memmove...
   for (int i=0; i<_size; i++) {
      _data[i] = rval->_data[i]; 
   }
}

ShuffleDeck::~ShuffleDeck()
{
   delete _data;
}

int ShuffleDeck::getNext(RNG& rng)
{
   if (_current == _size) {
      shuffle(rng);
   }
   return _data[_current++];

}

void ShuffleDeck::shuffle(RNG& rng)
{
   int picked;
   int tmp;
   for (int i=_size-1; i > 0; i--) {
//      picked = _random.irandom32(0, i);
     picked = irandom(0, i,rng);
      tmp = _data[i];
      _data[i] = _data[picked];
      _data[picked] = tmp;
   }
   _current = 0;
}
