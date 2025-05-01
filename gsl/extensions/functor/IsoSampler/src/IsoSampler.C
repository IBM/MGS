// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
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

void IsoSampler::duplicate(std::unique_ptr<IsoSampler>&& dup) const
{
   dup.reset(new IsoSampler(*this));
}

void IsoSampler::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new IsoSampler(*this));
}

void IsoSampler::duplicate(std::unique_ptr<CG_IsoSamplerBase>&& dup) const
{
   dup.reset(new IsoSampler(*this));
}

