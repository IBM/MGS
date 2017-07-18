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

#include "Lens.h"
#include "IsoSampler.h"
#include "CG_IsoSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include "LensContext.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "NodeSet.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "VectorOstream.h"
#include "NodeDescriptor.h"
#include <memory>
#include <vector>
#include <list>
#include <cmath>
#include "VectorOstream.h"

void IsoSampler::userInitialize(LensContext* CG_c) 
{
}

void IsoSampler::userExecute(LensContext* CG_c) 
{
  ConnectionContext *cc = CG_c->connectionContext;
  
  if (_done) {
    cc->done=true;
    _done=false;
    return;
  }
  
  if (cc->restart) {
    _done=false;
    cc->destinationSet->getNodes(_dstNodes); 
    cc->sourceSet->getNodes(_srcNodes); 

    _nbrNodes=_dstNodes.size();
    assert(_srcNodes.size()==_dstNodes.size());
    _nodeIndex=0;
  } // end of if (cc->restart)

  cc->sourceNode=_srcNodes[_nodeIndex];
  cc->destinationNode=_dstNodes[_nodeIndex];
  
  //assert(cc->sourceNode->getIndex()==cc->destinationNode->getIndex());
  //assert(cc->sourceNode->getNodeIndex()==cc->destinationNode->getNodeIndex());
			 
  if (++_nodeIndex==_nbrNodes) _done=true;

  cc->done = false;
}

IsoSampler::IsoSampler() 
   : CG_IsoSamplerBase(),
     _done(false),
     _nbrNodes(0),
     _nodeIndex(0)
{
}

IsoSampler::~IsoSampler() 
{
}

void IsoSampler::duplicate(std::auto_ptr<IsoSampler>& dup) const
{
   dup.reset(new IsoSampler(*this));
}

void IsoSampler::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new IsoSampler(*this));
}

void IsoSampler::duplicate(std::auto_ptr<CG_IsoSamplerBase>& dup) const
{
   dup.reset(new IsoSampler(*this));
}

