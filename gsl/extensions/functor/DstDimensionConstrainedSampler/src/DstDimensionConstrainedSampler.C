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
#include "DstDimensionConstrainedSampler.h"
#include "CG_DstDimensionConstrainedSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include "LensContext.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "NodeSet.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "VectorOstream.h"
#include "VolumeOdometer.h"
#include "NodeDescriptor.h"
#include <memory>
#include <vector>
#include <list>
#include <cmath>
#include "VectorOstream.h"

void DstDimensionConstrainedSampler::userInitialize(LensContext* CG_c, int& constrainedDstDim) 
{
  _constrainedDstDim=constrainedDstDim;
}

void DstDimensionConstrainedSampler::userExecute(LensContext* CG_c) 
{
  ConnectionContext *cc = CG_c->connectionContext;
  
  if (_done) {
    cc->done=true;
    _done=false;
    return;
  }
  
  if (cc->restart) {
    std::vector<int> dstBegin=cc->destinationSet->getBeginCoords();
    std::vector<int> dstEnd=cc->destinationSet->getEndCoords();
    std::vector<int> srcBegin=cc->sourceSet->getBeginCoords();
    std::vector<int> srcEnd=cc->sourceSet->getEndCoords();
    
    unsigned sz=dstBegin.size();
    assert(sz==srcBegin.size());

    _shortSrcDim=-1;
    for (unsigned i=0; i<sz; ++i) {
      if (srcEnd[i]-srcBegin[i]==0) {
	if (_shortSrcDim!=-1) {
	  std::cerr<<"Error in functor DstDimensionConstrainedSampler!"
		   <<std::endl<<"Source nodeset must have precisely one dimension of size 1."<<std::endl;
	  exit(0);
	}
	_shortSrcDim=i;
      }
    }
    if (_shortSrcDim==-1) {
      std::cerr<<"Error in functor DstDimensionConstrainedSampler!"
	       <<std::endl<<"Source nodeset must have precisely one dimension of size 1."<<std::endl;
      exit(0);
    }

    for (unsigned i=0; i<sz; ++i) {
      if (srcBegin[i]<dstBegin[i] || srcEnd[i]>dstEnd[i]) {
	std::cerr<<"Error in functor DstDimensionConstrainedSampler!"
		 <<std::endl<<"Source nodeset dimensions must not extend beyond destination nodeset dimensions."<<std::endl;
	exit(0);
      }
    }
  
    _beginConstrainedDimDst=dstBegin[_constrainedDstDim];
    _endConstrainedDimDst=dstEnd[_constrainedDstDim];

    cc->sourceSet->getNodes(_srcNodes); 
    _nbrNodesSrc=_srcNodes.size();
    _srcNodeIndex=0;

    _next=true;

  } // end of if (cc->restart)
  
  
  if (_next) {

    cc->sourceNode=_srcNodes[_srcNodeIndex];
    _currentConstrainedDimOffsetDst = 0;

    NodeSet sns(*cc->destinationSet);
    std::vector<int> srcCds;
    _srcNodes[_srcNodeIndex]->getNodeCoords(srcCds);

    std::vector<int> dstBegin=srcCds;
    std::vector<int> dstEnd=srcCds;

    dstBegin[_shortSrcDim]=dstBegin[_constrainedDstDim];
    dstEnd[_shortSrcDim]=dstEnd[_constrainedDstDim];

    dstBegin[_constrainedDstDim]=_beginConstrainedDimDst;
    dstEnd[_constrainedDstDim]=_endConstrainedDimDst; 
    
    sns.setCoords(dstBegin, dstEnd);
    sns.getNodes(_dstNodes);

    _nbrNodesDst = _dstNodes.size();

    _next=false;
  }
  
  cc->destinationNode=_dstNodes[_currentConstrainedDimOffsetDst];
  
  if (++_currentConstrainedDimOffsetDst==_nbrNodesDst) {
    _next=true;
    if (++_srcNodeIndex==_nbrNodesSrc) _done = true;
  }
  cc->done = false;
}

DstDimensionConstrainedSampler::DstDimensionConstrainedSampler() 
   : CG_DstDimensionConstrainedSamplerBase(),
     _constrainedDstDim(0),
     _shortSrcDim(0),
     _currentConstrainedDimOffsetDst(0),
     _beginConstrainedDimDst(0),
     _endConstrainedDimDst(0),
     _done(false),
     _nbrNodesDst(0),
     _nbrNodesSrc(0),
     _srcNodeIndex(0),
     _next(false)
{
}

DstDimensionConstrainedSampler::~DstDimensionConstrainedSampler() 
{
}

void DstDimensionConstrainedSampler::duplicate(std::unique_ptr<DstDimensionConstrainedSampler>& dup) const
{
   dup.reset(new DstDimensionConstrainedSampler(*this));
}

void DstDimensionConstrainedSampler::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new DstDimensionConstrainedSampler(*this));
}

void DstDimensionConstrainedSampler::duplicate(std::unique_ptr<CG_DstDimensionConstrainedSamplerBase>& dup) const
{
   dup.reset(new DstDimensionConstrainedSampler(*this));
}

