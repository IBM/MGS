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
