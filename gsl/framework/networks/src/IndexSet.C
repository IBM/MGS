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

#include "IndexSet.h"

IndexSet::IndexSet(std::vector<int>& begin, std::vector<int>& end)
: _begin(begin), _end(end)
{
}


IndexSet::IndexSet(const IndexSet& is)
: _begin(is._begin), _end(is._end)
{
}


IndexSet& IndexSet::operator=(const IndexSet& is)
{
   _begin = is._begin;
   _end = is._end;
   return (*this);
}


std::vector<int>& IndexSet::getBeginCoords()
{
   return _begin;
}


std::vector<int>& IndexSet::getEndCoords()
{
   return _end;
}


IndexSet::~IndexSet()
{

}
