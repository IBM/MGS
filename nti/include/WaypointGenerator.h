// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
