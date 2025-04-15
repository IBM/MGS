// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
