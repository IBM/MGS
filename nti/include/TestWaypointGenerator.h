// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TESTWAYPOINTGENERATOR_H
#define TESTWAYPOINTGENERATOR_H

#include <string>
#include "ShallowArray.h"

#include "WaypointGenerator.h"

class TestWaypointGenerator : public WaypointGenerator {
  public:
  TestWaypointGenerator() : Y(-155.0) {}

  void readWaypoints(std::vector<std::string>&, int) {}

  void next(ShallowArray<ShallowArray<double> >& waypointCoords, int nid) {
    waypointCoords.clear();
    waypointCoords.increaseSizeTo(3);
    if (Y > -1000.0) {
      ShallowArray<double> wp;
      wp.push_back(-875.0);
      wp.push_back(Y -= 100.0);
      wp.push_back(-116.0);
      waypointCoords[0] = wp;
      if (Y < -500.0) {
        ShallowArray<double> wp2;
        wp2.push_back(-1075.0);
        wp2.push_back(Y);
        wp2.push_back(-116.0);
        waypointCoords[1] = wp2;
        if (Y < -750.0) {
          ShallowArray<double> wp3;
          wp3.push_back(-1275.0);
          wp3.push_back(Y);
          wp3.push_back(-116.0);
          waypointCoords[2] = wp3;
        }
      }
    }
  }
  void reset() { Y = -155.0; }

  ~TestWaypointGenerator() {}

  private:
  double Y;
};
#endif
