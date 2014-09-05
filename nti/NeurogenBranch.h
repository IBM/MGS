// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

// Created by Heraldo Memelli
// summer 2012

#ifndef NEUROGENBRANCH_H
#define NEUROGENBRANCH_H

#include <math.h>
#include "ShallowArray.h"

class NeurogenSegment;
class NeurogenParams;

class NeurogenBranch {
 public:
  NeurogenBranch();
  virtual ~NeurogenBranch();

  int getNrSegments();
  double getLength();
  double getTotalSegmentLength();
  void addSegment(NeurogenSegment* s);
  void terminateBranch();
  void setSegmentIDs(NeurogenParams* params_p, int& count);
  void writeToSWC(std::ofstream& fout);
  void reset();
  NeurogenSegment* getFirstSegment();
  NeurogenSegment* getLastSegment();
  bool isBranching();
  ShallowArray<double>& getWaypoint1() {return Waypoint1;}
  ShallowArray<double>& getWaypoint2() {return Waypoint2;}
  NeurogenSegment* getSoma() {return soma;}
  void addWaypoint(ShallowArray<double>& wp);
  void setWaypoint1(ShallowArray<double>& wp) {Waypoint1=wp;}
  void setWaypoint2(ShallowArray<double>& wp) {Waypoint2=wp;}
  void setSoma(NeurogenSegment* s) {soma=s;}
  void clearWaypoints() {Waypoint1.clear(); Waypoint2.clear();}

 private:
  ShallowArray<NeurogenSegment*> BSegments;
  ShallowArray<double> Waypoint1, Waypoint2;
  NeurogenSegment* soma;
};


#endif /* BRANCH_H_ */
