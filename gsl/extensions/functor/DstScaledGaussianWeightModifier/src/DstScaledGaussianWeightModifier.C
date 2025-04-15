// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DstScaledGaussianWeightModifier.h"
#include "CG_DstScaledGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "Node.h"
#include "ConnectionContext.h"
#include <memory>
#include <vector>
#include <cmath>
#include "Grid.h"
#include "GridLayerDescriptor.h"

void DstScaledGaussianWeightModifier::userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max)
{
  _sigma=sigma;
  _max=max;
}

std::unique_ptr<ParameterSet> DstScaledGaussianWeightModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "DstScaledGaussianWeightModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1 = cc->sourceNode;
   NodeDescriptor* n2 = cc->destinationNode;
   assert(n1);
   assert(n2);

   int gridx1, gridy1, gridx2, gridy2;
   n1->getGridLayerDescriptor()->getGrid()->getSize2Dim(gridx1, gridy1);
   n2->getGridLayerDescriptor()->getGrid()->getSize2Dim(gridx2, gridy2);

   float centerx1 = float(gridx1-1)/2.0;
   float centery1 = float(gridy1-1)/2.0;
   float centerx2 = float(gridx2-1)/2.0;
   float centery2 = float(gridy2-1)/2.0;

   float scalex = float(gridx1)/float(gridx2);
   float scaley = float(gridy1)/float(gridy2);

   int x1, y1, x2, y2;
   n1->getNodeCoords2Dim(x1,y1);
   n2->getNodeCoords2Dim(x2,y2);

   float distancex=(float(x1)-centerx1)-((float(x2)-centerx2)*scalex);
   float distancey=(float(y1)-centery1)-((float(y2)-centery2)*scaley);
   float distance=distancex*distancex+distancey*distancey; 
   float weight = _max*exp(-distance/(2.0*_sigma*_sigma));

   NDPairList ndpl;

   std::string name="weight";
   FloatDataItem* fdi=new FloatDataItem(weight);
   std::unique_ptr<DataItem> fdi_ap(fdi);
   NDPair* ndp = new NDPair(name, fdi_ap);
   ndpl.push_back(ndp);

   name="distance";
   FloatDataItem* fdi2=new FloatDataItem(sqrt(distance));
   std::unique_ptr<DataItem> fdi_ap2(fdi2);
   NDPair* ndp2 = new NDPair(name, fdi_ap2);
   ndpl.push_back(ndp2);

   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(std::move(pset));
   pset->set(ndpl);
   return pset;
}

DstScaledGaussianWeightModifier::DstScaledGaussianWeightModifier() 
   : CG_DstScaledGaussianWeightModifierBase()
{
}

DstScaledGaussianWeightModifier::~DstScaledGaussianWeightModifier() 
{
}

void DstScaledGaussianWeightModifier::duplicate(std::unique_ptr<DstScaledGaussianWeightModifier>&& dup) const
{
   dup.reset(new DstScaledGaussianWeightModifier(*this));
}

void DstScaledGaussianWeightModifier::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new DstScaledGaussianWeightModifier(*this));
}

void DstScaledGaussianWeightModifier::duplicate(std::unique_ptr<CG_DstScaledGaussianWeightModifierBase>&& dup) const
{
   dup.reset(new DstScaledGaussianWeightModifier(*this));
}

