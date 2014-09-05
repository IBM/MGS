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
// ================================================================

/*
 * TouchVector.cpp
 *
 *  Created on: Feb 10, 2010
 *      Author: wagnerjo
 */

#include <list>
#include <algorithm>

#include "TouchVector.h"

void TouchVector::clear() {
  std::map<int, std::list<TouchIndex> >::iterator mapIterator = _touchMap.begin(), mapEnd=_touchMap.end();
  for (; mapIterator != mapEnd; ++mapIterator) mapIterator->second.clear();
  _touchMap.clear();
  BlockVector<Touch>::clear();
  assert(getBlockCount() == 0);
  assert(begin() == end());
}

void TouchVector::mapTouch(int i, TouchIndex ti) {
  assert (0 <= ti.getBlock() && ti.getBlock() < getBlockCount());
  assert(0 <= ti.getIndex() && ti.getIndex() < getValue(ti.getBlock())->getCount());
  std::map<int, std::list<TouchIndex> >::iterator mapIter = _touchMap.find(i);
  if (mapIter==_touchMap.end()) {
    std::list<TouchIndex> newList;
    newList.push_back(ti);
    _touchMap[i]=newList;
  } else {
    std::list<TouchIndex>& touchList=mapIter->second;
    std::list<TouchIndex>::iterator titer=find(touchList.begin(), touchList.end(), ti);
    if (titer==touchList.end()) touchList.push_back(ti);
  }
}
TouchVector::TouchIterator TouchVector::begin() {
  return(TouchIterator(this, 0, 0));
}

TouchVector::TouchIterator TouchVector::end() {
  return(TouchIterator(this, getBlockCount(), 0));
}

TouchVector::TouchIterator TouchVector::begin(Capsule &capsule, int direction) {
  double (Touch::*funptr) () = ( (direction==0) ? &Touch::getKey1 : &Touch::getKey2 );
  TouchIterator i =  begin(), I = end();
  if (fieldLastEnd != I && (*fieldLastEnd.*funptr)() == capsule.getKey()) return(fieldLastEnd);
  for (; i != I; ++i) {
    if ( (*i.*funptr)() == capsule.getKey()) return(i);
  }
  return(I);
}

TouchVector::TouchIterator TouchVector::end(Capsule &capsule, int direction) {
  double (Touch::*funptr) () = ( (direction==0) ? &Touch::getKey1 : &Touch::getKey2 );
  TouchIterator i =  begin(capsule, direction), I = end();
  for (; i != I; ++i) {
    if ( (*i.*funptr)() != capsule.getKey())  {
      fieldLastEnd = i;
      return(i);
    }
  }
  fieldLastEnd = I;
  return(I);
}

void TouchVector::sort(Touch::compare &c) {
  heapSort(c);
}

void TouchVector::heapSort(Touch::compare &c)
{
  int i, size=getCount();
  Touch temp;
  TouchVector& v=(*this);

  for (i=size-1; i>=0; --i)
    demote(c, i, size-1);

  for (i=size-1; i>=1; --i) {
    temp=v[0];
    v[0]=v[i];
    v[i]=temp;
    demote(c,0,i-1);
  }
}

void TouchVector::demote(Touch::compare &c, int boss, int bottomEmployee)
{
  TouchVector& v=(*this);
  int topEmployee;
  Touch temp;
  while (bottomEmployee>=2*boss) {
    topEmployee = 2*boss + ( ( (bottomEmployee!=2*boss) && (!c(v[2*boss+1],v[2*boss]) ) ) ? 1 : 0 );
    assert(topEmployee<getCount());
    if (c(v[boss],v[topEmployee])) {
      temp=v[boss];
      v[boss]=v[topEmployee];
      v[topEmployee]=temp;
      boss=topEmployee;
    }
    else break;
  }
}

void TouchVector::bubbleSort(Touch::compare &c)
{
  for (TouchIterator i =  begin(), I = end(); i != I; ++i) {
    for (TouchIterator j = i, J = end(); j != J; ++j) {
      if (!c(*i, *j)) {
	Touch temp = *i; *i = *j; *j = temp;
      }
    }
  }
}

bool TouchVector::unique()
{
  bool rval=true;
  int L=getBlockCount()-1;
  for (TouchIterator i =  begin(), I = end(); i != I; ++i) {
    for (TouchIterator j = i, J = end(); rval && j != J; ++j) {
      if (i!=j && (*i)==(*j)) {
	rval=false;
      }
    }
  }
  return rval;
}
