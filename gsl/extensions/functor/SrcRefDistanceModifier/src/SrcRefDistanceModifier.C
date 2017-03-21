// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "SrcRefDistanceModifier.h"
#include "CG_SrcRefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <memory>
#include <cmath>

void SrcRefDistanceModifier::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

std::auto_ptr<ParameterSet> SrcRefDistanceModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

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
   std::auto_ptr<DataItem> fdi_ap(fdi);

   NDPair* ndp = new NDPair(name, fdi_ap);
   NDPairList ndpl;
   ndpl.push_back(ndp);
   std::auto_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(pset);
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

void SrcRefDistanceModifier::duplicate(std::auto_ptr<SrcRefDistanceModifier>& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

void SrcRefDistanceModifier::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

void SrcRefDistanceModifier::duplicate(std::auto_ptr<CG_SrcRefDistanceModifierBase>& dup) const
{
   dup.reset(new SrcRefDistanceModifier(*this));
}

