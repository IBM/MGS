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

#include "DstRefDistanceModifier.h"
#include "CG_DstRefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <memory>
#include <cmath>

void DstRefDistanceModifier::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

std::unique_ptr<ParameterSet> DstRefDistanceModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "DstRefDistanceModifier: functor did not return a Parameter Set");
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

   std::string name="distance";
   FloatDataItem* fdi=new FloatDataItem(distance);
   std::unique_ptr<DataItem> fdi_ap(fdi);

   NDPair* ndp = new NDPair(name, fdi_ap);
   NDPairList ndpl;
   ndpl.push_back(ndp);
   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(pset);
   pset->set(ndpl);
   return pset;
}

DstRefDistanceModifier::DstRefDistanceModifier() 
   : CG_DstRefDistanceModifierBase()
{
}

DstRefDistanceModifier::~DstRefDistanceModifier() 
{
}

void DstRefDistanceModifier::duplicate(std::unique_ptr<DstRefDistanceModifier>& dup) const
{
   dup.reset(new DstRefDistanceModifier(*this));
}

void DstRefDistanceModifier::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new DstRefDistanceModifier(*this));
}

void DstRefDistanceModifier::duplicate(std::unique_ptr<CG_DstRefDistanceModifierBase>& dup) const
{
   dup.reset(new DstRefDistanceModifier(*this));
}

