#include "Lens.h"
#include "IsoSamplerHybrid.h"
#include "CG_IsoSamplerHybridBase.h"
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

void IsoSamplerHybrid::userInitialize(LensContext* CG_c) 
{
}

void IsoSamplerHybrid::userExecute(LensContext* CG_c) 
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
  
  //NOTE: relax this constraint
  //assert(cc->sourceNode->getIndex()==cc->destinationNode->getIndex());
  //assert(cc->sourceNode->getNodeIndex()==cc->destinationNode->getNodeIndex());
			 
  if (++_nodeIndex==_nbrNodes) _done=true;

  cc->done = false;
}

IsoSamplerHybrid::IsoSamplerHybrid() 
   : CG_IsoSamplerHybridBase(),
     _done(false),
     _nbrNodes(0),
     _nodeIndex(0)
{
}

IsoSamplerHybrid::~IsoSamplerHybrid() 
{
}

void IsoSamplerHybrid::duplicate(std::unique_ptr<IsoSamplerHybrid>&& dup) const
{
   dup.reset(new IsoSamplerHybrid(*this));
}

void IsoSamplerHybrid::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new IsoSamplerHybrid(*this));
}

void IsoSamplerHybrid::duplicate(std::unique_ptr<CG_IsoSamplerHybridBase>&& dup) const
{
   dup.reset(new IsoSamplerHybrid(*this));
}

