// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SrcRefDistanceModifier.h"
#include "CG_SrcRefDistanceModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <memory>
#include <cmath>

void SrcRefDistanceModifier::userInitialize(GslContext* CG_c, Functor*& f) 
{
}

std::unique_ptr<ParameterSet> SrcRefDistanceModifier::userExecute(GslContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "SrcRefDistanceModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1 = cc->sourceNode;
   NodeDescriptor* n2 = cc->sourceRefNode;

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
   psdi->getParameterSet()->duplicate(std::move(pset));
   pset->set(ndpl);
   return pset;
}

SrcRefDistanceModifier::SrcRefDistanceModifier() 
   : CG_SrcRefDistanceModifierBase()
{
}

SrcRefDistanceModifier::~SrcRefDistanceModifier() 
{
}

void SrcRefDistanceModifier::duplicate(std::unique_ptr<SrcRefDistanceModifier>&& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

void SrcRefDistanceModifier::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

void SrcRefDistanceModifier::duplicate(std::unique_ptr<CG_SrcRefDistanceModifierBase>&& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

