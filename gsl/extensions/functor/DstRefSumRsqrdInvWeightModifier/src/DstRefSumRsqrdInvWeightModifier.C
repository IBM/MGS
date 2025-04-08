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

#include "DstRefSumRsqrdInvWeightModifier.h"
#include "CG_DstRefSumRsqrdInvWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "Node.h"
#include "ConnectionContext.h"
#include <memory>
#include <list>
#include <cmath>

void DstRefSumRsqrdInvWeightModifier::userInitialize(LensContext* CG_c, Functor*& f, int& maxDim, bool& setDistance) 
{
  assert(maxDim>0);
  _setDistance=setDistance;

  std::list<float> radii;
  _maxDistance = int(ceil(sqrt(2.0*maxDim*maxDim)));
  for (int coords0=-_maxDistance; coords0<=_maxDistance; ++coords0) {
    for (int coords1=-_maxDistance; coords1<=_maxDistance; ++coords1) {
      float radius=sqrt(double(coords0*coords0+coords1*coords1));
      radii.push_back(radius);
    }
  }
  radii.sort();
  std::list<float> nsi;
  std::list<float>::iterator iter=radii.begin();
  std::list<float>::iterator end=radii.end();  
  float prevR=*iter;
  ++iter;
  int n=0;
  for (; iter!=end; ++iter) {
    ++n;
    while (*iter==prevR) {
      ++n;
      prevR=*iter;
      ++iter;
      if (iter==end) break;
    }
    if (iter==end) break;
    nsi.push_back((*iter-prevR)/float(n));
    prevR=*iter;
  }
  radii.unique();
  iter=radii.end();
  --iter;
  std::list<float>::iterator iter2=nsi.end();
  std::list<float>::iterator end2=nsi.begin();
  do {
    --iter;
    --iter2;
  } while (*iter>_maxDistance);
  float sumNsi=0.0090287;
  for (--end2; iter2!=end2; --iter, --iter2) {
    sumNsi+=*iter2;
    _radiusMap[*iter]=sumNsi;
    //std::cerr<<*iter<<" : "<<sumNsi<<std::endl;
  }
}

std::unique_ptr<ParameterSet> DstRefSumRsqrdInvWeightModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "DstRefSumRsqrdInvWeightModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1 = cc->destinationNode;
   NodeDescriptor* n2 = cc->destinationRefNode;

   assert(n1);
   assert(n2);

   float distance=0;
   int cds1X, cds1Y, cds2X, cds2Y;
   n1->getNodeCoords2Dim(cds1X, cds1Y);
   n2->getNodeCoords2Dim(cds2X, cds2Y);
   distance+=(cds1X-cds2X)*(cds1X-cds2X);
   distance+=(cds1Y-cds2Y)*(cds1Y-cds2Y);
   distance=sqrt(distance);
   assert(distance<_maxDistance);
   float weight = _radiusMap[distance];

   NDPairList ndpl;

   std::string name="weight";
   FloatDataItem* fdi=new FloatDataItem(weight);
   std::unique_ptr<DataItem> fdi_ap(fdi);
   NDPair* ndp = new NDPair(name, fdi_ap);
   ndpl.push_back(ndp);

   if (_setDistance) {
     name="distance";
     FloatDataItem* fdi2=new FloatDataItem(distance);
     std::unique_ptr<DataItem> fdi_ap2(fdi2);
     NDPair* ndp2 = new NDPair(name, fdi_ap2);
     ndpl.push_back(ndp2);
   }

   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(std::move(pset));
   pset->set(ndpl);
   return pset;
}

DstRefSumRsqrdInvWeightModifier::DstRefSumRsqrdInvWeightModifier() 
   : CG_DstRefSumRsqrdInvWeightModifierBase()
{
}

DstRefSumRsqrdInvWeightModifier::~DstRefSumRsqrdInvWeightModifier() 
{
}

void DstRefSumRsqrdInvWeightModifier::duplicate(std::unique_ptr<DstRefSumRsqrdInvWeightModifier>&& dup) const
{
   dup.reset(new DstRefSumRsqrdInvWeightModifier(*this));
}

void DstRefSumRsqrdInvWeightModifier::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new DstRefSumRsqrdInvWeightModifier(*this));
}

void DstRefSumRsqrdInvWeightModifier::duplicate(std::unique_ptr<CG_DstRefSumRsqrdInvWeightModifierBase>&& dup) const
{
   dup.reset(new DstRefSumRsqrdInvWeightModifier(*this));
}

