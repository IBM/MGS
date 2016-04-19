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

//#define SOMA_BRANCH 0
//#define APICAL_BRANCH 1
//#define BASAL_BRANCH 2
//#define AXON_BRANCH 3
//#define AIS_BRANCH 4
//#define TUFT_BRANCH 5

class Neuron;

class Branch {
   public:
		 //AIS = axon initiation site
		 //TUFTEDDEN = the distal tufted dendritic part
	  enum BranchType { _SOMA=0, _AXON=1, _BASALDEN=2, _APICALDEN=3, _AIS=4, _TUFTEDDEN=5, _BOUTON=6 };
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
      int _branchType; // NOTE: Follow .SWC convention minus 1, e.g. soma has branchType=0
      int _branchOrder; // NOTE: soma has branchOrder = 1, and it increases at each branching
      double _dist2Soma; // along-fiber-distance to the soma from the first Segment of the branch
      int _numberOfSegments;
      //int _numberOfResampledSegments;
      int _branchIndex; // order of the Branch in the associated neuron '_neuron' starting from 0 = soma
	  //NOTE: A soma is a 2-segment branch with both having the same (x,y,z,radius)
      
      Segment* _segments;
      Neuron* _neuron;
      Segment* _rootSegment; // reference to the root segment (i.e. the soma) of the neuron
      int _resampledTerminalIndex; // the new index of the last segment in that branch after resampling
      double _displacedTerminalCoords[3];
};
#endif
