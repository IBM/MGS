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

#include "AntiredundancyTouchFilter.h"
#include "Touch.h"
#include "TouchAnalyzer.h"
#include "Branch.h"
#include "Segment.h"
#include "Tissue.h"
#include "TouchAggregator.h"

#include <cassert>
#include <algorithm>
#include <list>

AntiredundancyTouchFilter::AntiredundancyTouchFilter(
						     TouchAggregator* touchAggregator,
						     Tissue* tissue)
  :  _touchAggregator(touchAggregator),
     _tissue(tissue),
     _touchAnalyzer(0),
     _neuronMask(0),
     _branchMask(0),
     _segmentMask(0)
{
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  maskVector.push_back(SegmentDescriptor::neuronIndex);
  _neuronMask = _segmentDescriptor.getMask(maskVector);
  
  maskVector.push_back(SegmentDescriptor::branchIndex);
  _branchMask = _segmentDescriptor.getMask(maskVector);
  
  maskVector.push_back(SegmentDescriptor::segmentIndex);
  _segmentMask = _segmentDescriptor.getMask(maskVector);
}

void AntiredundancyTouchFilter::filterTouches()
{
#ifndef LTWT_TOUCH
  assert(_touchAggregator);
  assert(_tissue);
  int numTouches = _touchAggregator->getNumberOfTouches();

  if (!_tissue->isEmpty() && numTouches!=0) {  

    Segment *segmentsBegin = _tissue->getSegments(),
      *segmentsEnd = segmentsBegin+_tissue->getSegmentArraySize();
    for (Segment* segmentPtr=segmentsBegin; segmentPtr!=segmentsEnd; ++segmentPtr)
      segmentPtr->resetTouches();
    
    Touch* touches = _touchAggregator->getTouches();
    Touch* touchEnd=touches+numTouches;

    std::sort(touches, touchEnd, Touch::compare(0));
    eliminateRedundantTouches(touches, touchEnd, 0);
    std::sort(touches, touchEnd, Touch::compare(1));
    //eliminateRedundantTouches(touches, touchEnd, 1);
    
    double s1Key, s2Key;
    int neuronIndex, branchIndex, segmentIndex;
    Neuron* neurons = _tissue->getNeurons();

    assert(_touchAnalyzer);
  
    for (Touch* touchPtr=touches; touchPtr!=touchEnd; ++touchPtr) {
      if (touchPtr->remains()) {
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
  }
  else assert(numTouches==0);
#endif
}

void AntiredundancyTouchFilter::eliminateRedundantTouches(Touch* touchStart, Touch* touchEnd, int c0)
{
#ifndef LTWT_TOUCH
  Touch *touchOrigin, *touchPtr=touchStart;
  int c1=1;  
  int endTouchIdxAv=2;
  int endTouchIdxBv=3;
  if (c0==1) {
    c1=0;
    endTouchIdxAv=0;
    endTouchIdxBv=1;
  }

  double s1u, s0u=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c0], _segmentMask);
  unsigned int s1v, s0v=_segmentDescriptor.getSegmentIndex(touchPtr->getKeys()[c1]);
  double b1v, b0v=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c1], _branchMask);

  bool endTouch1Av, endTouch0Av=(touchPtr->getEndTouches()[endTouchIdxAv]==1);
  bool endTouch1Bv, endTouch0Bv=(touchPtr->getEndTouches()[endTouchIdxBv]==1);
  
  while (++touchPtr!=touchEnd) {
    s1u=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c0], _segmentMask);
    s1v=_segmentDescriptor.getSegmentIndex(touchPtr->getKeys()[c1]);
    b1v=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c1], _branchMask);
    endTouch1Av=(touchPtr->getEndTouches()[endTouchIdxAv]==1);
    endTouch1Bv=(touchPtr->getEndTouches()[endTouchIdxBv]==1);
    if (s1u==s0u && b1v==b0v && s1v==s0v+1 && endTouch0Bv && endTouch1Av) {
      touchOrigin=touchPtr-1;
      touchOrigin->eliminate();
      while (s1u==s0u && b1v==b0v && s1v==++s0v && endTouch1Av) {
	touchPtr->eliminate();
	if (touchPtr->getDistance() < touchOrigin->getDistance()) touchOrigin=touchPtr;
	if (++touchPtr==touchEnd) break;
	s1u=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c0], _segmentMask);
	s1v=_segmentDescriptor.getSegmentIndex(touchPtr->getKeys()[c1]);
	b1v=_segmentDescriptor.getSegmentKey(touchPtr->getKeys()[c1], _branchMask);
	endTouch1Av=(touchPtr->getEndTouches()[endTouchIdxAv]==1);
	if (!endTouch1Bv) {
	  endTouch1Bv=(touchPtr->getEndTouches()[endTouchIdxBv]==1);
	  break;
	}
	endTouch1Bv=(touchPtr->getEndTouches()[endTouchIdxBv]==1);
      }

      touchOrigin->reinstate();
      if (touchPtr==touchEnd) break;
    }

    s0u=s1u;
    s0v=s1v;
    b0v=b1v;
    endTouch0Av=endTouch1Av;
    endTouch0Bv=endTouch1Bv;
  }
#endif
}

void AntiredundancyTouchFilter::setTouchAnalyzer(TouchAnalyzer* touchAnalyzer)
{
  _touchAnalyzer=touchAnalyzer;
}
	

AntiredundancyTouchFilter::~AntiredundancyTouchFilter()
{
}
