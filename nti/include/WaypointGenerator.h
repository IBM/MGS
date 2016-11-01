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

#ifndef WAYPOINTGENERATOR_H
#define WAYPOINTGENERATOR_H

#include <string>
#include "ShallowArray.h"

class WaypointGenerator {
  public:
  WaypointGenerator() {}
  virtual void next(ShallowArray<ShallowArray<double> >& waypointCoords,
                    int nid) = 0;
  virtual void readWaypoints(std::vector<std::string>& fileNames,
                             int nrNeurons) = 0;
  virtual void reset() = 0;
  virtual ~WaypointGenerator() {}
};

#endif
