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
#include "SrcDimensionConstrainedSampler.h"
#include "CG_SrcDimensionConstrainedSamplerBase.h"
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

void SrcDimensionConstrainedSampler::userInitialize(LensContext* CG_c, int& constrainedSrcDim) 
{
  _constrainedSrcDim=constrainedSrcDim;
}

void SrcDimensionConstrainedSampler::userExecute(LensContext* CG_c) 
{
  ConnectionContext *cc = CG_c->connectionContext;
  
  if (_done) {
    cc->done=true;
    _done=false;
    return;
  }
  
  if (cc->restart) {
    std::vector<int> srcBegin=cc->sourceSet->getBeginCoords();
    std::vector<int> srcEnd=cc->sourceSet->getEndCoords();
    std::vector<int> dstBegin=cc->destinationSet->getBeginCoords();
    std::vector<int> dstEnd=cc->destinationSet->getEndCoords();
    
    unsigned sz=srcBegin.size();
    assert(sz==dstBegin.size());

    _shortDstDim=-1;
    for (unsigned i=0; i<sz; ++i) {
      if (dstEnd[i]-dstBegin[i]==0) {
	if (_shortDstDim!=-1) {
	  std::cerr<<"Error in functor SrcDimensionConstrainedSampler!"
		   <<std::endl<<"Destination nodeset must have precisely one dimension of size 1."<<std::endl;
	  exit(0);
	}
	_shortDstDim=i;
      }
    }
    if (_shortDstDim==-1) {
      std::cerr<<"Error in functor SrcDimensionConstrainedSampler!"
	       <<std::endl<<"Destination nodeset must have precisely one dimension of size 1."<<std::endl;
      exit(0);
    }

    for (unsigned i=0; i<sz; ++i) {
      if (dstBegin[i]<srcBegin[i] || dstEnd[i]>srcEnd[i]) {
	std::cerr<<"Error in functor SrcDimensionConstrainedSampler!"
		 <<std::endl<<"Destination nodeset dimensions must not extend beyond source nodeset dimensions."<<std::endl;
	exit(0);
      }
    }
  
    _beginConstrainedDimSrc=srcBegin[_constrainedSrcDim];
    _endConstrainedDimSrc=srcEnd[_constrainedSrcDim];

    cc->destinationSet->getNodes(_dstNodes); 
    _nbrNodesDst=_dstNodes.size();
    _dstNodeIndex=0;

    _next=true;

  } // end of if (cc->restart)
  
  
  if (_next) {

    cc->destinationNode=_dstNodes[_dstNodeIndex];
    _currentConstrainedDimOffsetSrc = 0;

    NodeSet sns(*cc->sourceSet);
    std::vector<int> dstCds;
    _dstNodes[_dstNodeIndex]->getNodeCoords(dstCds);

    std::vector<int> srcBegin=dstCds;
    std::vector<int> srcEnd=dstCds;

    srcBegin[_shortDstDim]=srcBegin[_constrainedSrcDim];
    srcEnd[_shortDstDim]=srcEnd[_constrainedSrcDim];

    srcBegin[_constrainedSrcDim]=_beginConstrainedDimSrc;
    srcEnd[_constrainedSrcDim]=_endConstrainedDimSrc; 
    
    sns.setCoords(srcBegin, srcEnd);
    sns.getNodes(_srcNodes);

    _nbrNodesSrc = _srcNodes.size();

    _next=false;
  }
  
  cc->sourceNode=_srcNodes[_currentConstrainedDimOffsetSrc];
  
  if (++_currentConstrainedDimOffsetSrc==_nbrNodesSrc) {
    _next=true;
    if (++_dstNodeIndex==_nbrNodesDst) _done = true;
  }
  cc->done = false;
}

SrcDimensionConstrainedSampler::SrcDimensionConstrainedSampler() 
   : CG_SrcDimensionConstrainedSamplerBase(),
     _constrainedSrcDim(0),
     _shortDstDim(0),
     _currentConstrainedDimOffsetSrc(0),
     _beginConstrainedDimSrc(0),
     _endConstrainedDimSrc(0),
     _done(false),
     _nbrNodesSrc(0),
     _nbrNodesDst(0),
     _dstNodeIndex(0),
     _next(false)
{
}

SrcDimensionConstrainedSampler::~SrcDimensionConstrainedSampler() 
{
}

void SrcDimensionConstrainedSampler::duplicate(std::unique_ptr<SrcDimensionConstrainedSampler>& dup) const
{
   dup.reset(new SrcDimensionConstrainedSampler(*this));
}

void SrcDimensionConstrainedSampler::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new SrcDimensionConstrainedSampler(*this));
}

void SrcDimensionConstrainedSampler::duplicate(std::unique_ptr<CG_SrcDimensionConstrainedSamplerBase>& dup) const
{
   dup.reset(new SrcDimensionConstrainedSampler(*this));
}

