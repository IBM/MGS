// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef BRANCH_H
#define BRANCH_H

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string.h>

#include "Segment.h"

class Neuron;

class Branch {
   public:
      Branch();
      Branch(Branch const & s);
      Branch& operator=(const Branch& branch);

      void resample(std::vector<Segment>& segments, double pointSpacing);
      void resetBranchRoots(std::vector<Segment>& segments);
      Segment* loadBinary(FILE*, Segment*, Neuron*, const int branchIndex);
      Segment* loadText(FILE*, Segment*, Neuron*, const int branchIndex, std::list<int>& branchTerminals,
			double xOffset, double yOffset, double zOffset, int cellBodyCorrection);
      void writeCoordinates(FILE*);
      const int getBranchType() {return _branchType;}
      const int getBranchOrder() {return _branchOrder;}
      const double getDist2Soma() {return _dist2Soma;}
      const double getLength();
      const int getBranchIndex() {return _branchIndex;}
      const int getNumberOfSegments() {return _numberOfSegments;}
      Segment* getSegments() {return _segments;}
      Segment* getTerminalSegment() {return &_segments[_numberOfSegments-1];}
      double* getDisplacedTerminalCoords() {return _displacedTerminalCoords;}
      void setDisplacedTerminalCoords();
      void resetSegments(Segment* segments);
      void resetSegments(Segment* segments, Segment* rootSegment);
      void resetBranchIndex(int branchIndex) {_branchIndex=branchIndex;}
      Segment* getRootSegment() {return _rootSegment;}
      void findRootSegment();
      void setBranchOrder(int branchOrder) {_branchOrder=branchOrder;}
      void setDist2Soma(double distance);
      Neuron* getNeuron() {return _neuron;}
      int getResampledTerminalIndex() {return _resampledTerminalIndex;}
      void resetCoordinates(Segment& newSeg, Segment* seg2, Segment* seg3, double d, double L, double pointSpacing);
      bool nextSegment(Segment* seg1, Segment*& seg2, int& i, double& L);
      bool isTerminalBranch();

   private:
      int _branchType;
      int _branchOrder;
      double _dist2Soma;
      int _numberOfSegments;
      int _numberOfResampledSegments;
      int _branchIndex;
      
      Segment* _segments;
      Neuron* _neuron;
      Segment* _rootSegment;
      int _resampledTerminalIndex;
      double _displacedTerminalCoords[3];
};
#endif
