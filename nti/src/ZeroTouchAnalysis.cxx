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

#include "ZeroTouchAnalysis.h"
#include "SegmentDescriptor.h"
#include "TouchTable.h"
#include "Tissue.h"
#include "Neuron.h"

#include <vector>

ZeroTouchAnalysis::ZeroTouchAnalysis(Tissue* tissue) :
	_tissue(tissue),
	_neuronPtr(_tissue->getNeurons()),
	_neuronEnd(_tissue->getNeurons()+_tissue->getNeuronArraySize())
{
  /*
   std::vector<SegmentDescriptor::SegmentKeyData> indexMask;
   indexMask.push_back(SegmentDescriptor::neuronIndex);
   indexMask.push_back(SegmentDescriptor::branchIndex);
   indexMask.push_back(SegmentDescriptor::segmentIndex);
   TouchTable* table = new TouchTable(indexMask);
   table->setOutput(false);
   _touchTables.push_back(table);
  */
}

bool ZeroTouchAnalysis::analyze(std::list<Translation>& translations, std::list<Rotation>& rotations)
{
  bool done=true;
  /*
  bool done=false;
  if (_touchTables[0]->getTouchCount()==0) done=true;
  //An example of how to rotate a neuron as a result of an analysis
  if (!_tissue->isEmpty()) {
    for (Neuron* neuronPtr=_neuronPtr; neuronPtr != _neuronEnd; ++neuronPtr) {
      Rotation r;
      r.setRotation(1.0);
      r.setIndex(neuronPtr->getNeuronIndex());
      rotations.push_back(r);
    }
  }
  */
  return done;
}

ZeroTouchAnalysis::~ZeroTouchAnalysis()
{
}
