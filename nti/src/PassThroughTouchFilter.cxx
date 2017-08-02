// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "PassThroughTouchFilter.h"
#include "Touch.h"
#include "TouchAnalyzer.h"
#include "Branch.h"
#include "Segment.h"
#include "Tissue.h"
#include "TouchAggregator.h"

#include <cassert>
#include <algorithm>
#include <list>

PassThroughTouchFilter::PassThroughTouchFilter(
					       TouchAggregator* touchAggregator,
					       Tissue* tissue)
  :  _touchAggregator(touchAggregator),
     _tissue(tissue),
     _touchAnalyzer(0)
{
}

void PassThroughTouchFilter::filterTouches()
{
  assert(_touchAggregator);
  assert(_tissue);
  int numTouches = _touchAggregator->getNumberOfTouches();

  if (!_tissue->isEmpty() && numTouches!=0) {  

    Segment *segmentsBegin = _tissue->getSegments(),
      *segmentsEnd = segmentsBegin+_tissue->getSegmentArraySize();
    for (Segment* segmentPtr=segmentsBegin; segmentPtr!=segmentsEnd; ++segmentPtr)
      segmentPtr->resetTouches();
    
    Touch* touchPtr = _touchAggregator->getTouches();
    Touch* touchEnd = touchPtr+_touchAggregator->getNumberOfTouches();

    double s1Key, s2Key;
    int neuronIndex, branchIndex, segmentIndex;
    Neuron* neurons = _tissue->getNeurons();

    assert(_touchAnalyzer);

    for (; touchPtr!=touchEnd; ++touchPtr) {
      _touchAnalyzer->evaluateTouch(*touchPtr);

      s1Key=touchPtr->getKey1();
      s2Key=touchPtr->getKey2();
						
      int neuronIndex = _tissue->getNeuronIndex(_segmentDescriptor.getNeuronIndex(s1Key));
      if (_tissue->isInTissue(neuronIndex)) {
	branchIndex=_segmentDescriptor.getBranchIndex(s1Key);
	segmentIndex=_segmentDescriptor.getSegmentIndex(s1Key);
	neurons[neuronIndex].getBranches()[branchIndex].getSegments()[segmentIndex].addTouch(touchPtr);
      }
					
      neuronIndex = _tissue->getNeuronIndex(_segmentDescriptor.getNeuronIndex(s2Key));
      if (_tissue->isInTissue(neuronIndex)) {
	branchIndex=_segmentDescriptor.getBranchIndex(s2Key);
	segmentIndex=_segmentDescriptor.getSegmentIndex(s2Key);
	neurons[neuronIndex].getBranches()[branchIndex].getSegments()[segmentIndex].addTouch(touchPtr);
      }
    }
  }
  else assert(numTouches==0);
}	

void PassThroughTouchFilter::setTouchAnalyzer(TouchAnalyzer* touchAnalyzer)
{
  _touchAnalyzer=touchAnalyzer;
}

PassThroughTouchFilter::~PassThroughTouchFilter()
{
}
