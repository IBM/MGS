// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GraphConnection.h"
#include <sstream>

GraphConnection::GraphConnection(unsigned graphId, float weight) 
   : _graphId(graphId), _weight(weight)
{
}

GraphConnection::GraphConnection() 
   : _graphId(0), _weight(0)
{
}

std::ostream& operator<<(std::ostream& os, const GraphConnection& inp)
{
   os << inp.getGraphId() << " " << inp.getWeight() << " ";
   return os;
}

std::istream& operator>>(std::istream& is, GraphConnection& inp)
{
  unsigned gid;
  is >> gid;
  inp.setGraphId(gid);
  float wt;
  is >> wt;
  inp.setWeight(wt);
  return is;
}
