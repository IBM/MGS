// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SrcRefGaussianWeightModifier.h"
#include "CG_SrcRefGaussianWeightModifierBase.h"
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

void SrcRefGaussianWeightModifier::userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance) 
{
  _sigma=sigma;
  _max=max;
  _wrapDistance=wrapDistance;
}

std::unique_ptr<ParameterSet> SrcRefGaussianWeightModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "SrcRefGaussianWeightModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1 = cc->sourceNode;
   NodeDescriptor* n2 = cc->sourceRefNode;

   assert(n1);
   assert(n2);


   /*
   float distance=0;
   int cds1X, cds1Y, cds2X, cds2Y;
   n1->getNodeCoords2Dim(cds1X, cds1Y);
   n2->getNodeCoords2Dim(cds2X, cds2Y);
   distance+=(cds1X-cds2X)*(cds1X-cds2X);
   distance+=(cds1Y-cds2Y)*(cds1Y-cds2Y);
   distance=sqrt(distance);
   */

   float distance=0, ddist;
   std::vector<int> coords1, coords2, gridSize;
   n1->getNodeCoords(coords1);
   n2->getNodeCoords(coords2);
   std::vector<int>::iterator i1, i2, i3, end1 = coords1.end();

   if (_wrapDistance==0){
      for (i1=coords1.begin(),i2=coords2.begin();i1!=end1;++i1, ++i2){
         ddist = *i1 - *i2;
         distance += ddist*ddist;
      }
      distance=sqrt(distance);
   } 
   else {
      std::vector<int> gridSize;
      Grid *g = n2->getGridLayerDescriptor()->getGrid();
      gridSize = g->getSize();
      for (i1=coords1.begin(),i2=coords2.begin(),i3=gridSize.begin();i1!=end1;++i1, ++i2, ++i3){
         ddist = std::abs(*i1 - *i2);
         if (ddist > *i3/2) ddist = *i3 - ddist;
         distance += ddist*ddist;
      }
      distance=sqrt(distance);
   }

   float weight = _max*exp(-(distance*distance)/(2.0*_sigma*_sigma));

   NDPairList ndpl;

   std::string name="weight";
   FloatDataItem* fdi=new FloatDataItem(weight);
   std::unique_ptr<DataItem> fdi_ap(fdi);
   NDPair* ndp = new NDPair(name, fdi_ap);
   ndpl.push_back(ndp);

   name="distance";
   FloatDataItem* fdi2=new FloatDataItem(distance);
   std::unique_ptr<DataItem> fdi_ap2(fdi2);
   NDPair* ndp2 = new NDPair(name, fdi_ap2);
   ndpl.push_back(ndp2);

   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(std::move(pset));
   pset->set(ndpl);
   return pset;
}

SrcRefGaussianWeightModifier::SrcRefGaussianWeightModifier() 
   : CG_SrcRefGaussianWeightModifierBase()
{
}

SrcRefGaussianWeightModifier::~SrcRefGaussianWeightModifier() 
{
}

void SrcRefGaussianWeightModifier::duplicate(std::unique_ptr<SrcRefGaussianWeightModifier>&& dup) const
{
   dup.reset(new SrcRefGaussianWeightModifier(*this));
}

void SrcRefGaussianWeightModifier::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new SrcRefGaussianWeightModifier(*this));
}

void SrcRefGaussianWeightModifier::duplicate(std::unique_ptr<CG_SrcRefGaussianWeightModifierBase>&& dup) const
{
   dup.reset(new SrcRefGaussianWeightModifier(*this));
}

