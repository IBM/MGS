// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ReversedSrcRefGaussianWeightModifier.h"
#include "CG_ReversedSrcRefGaussianWeightModifierBase.h"
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

void ReversedSrcRefGaussianWeightModifier::userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance) 
{
  _sigma=sigma;
  _max=max;
  _wrapDistance=wrapDistance;
}

std::auto_ptr<ParameterSet> ReversedSrcRefGaussianWeightModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "ReversedSrcRefGaussianWeightModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1 = cc->destinationNode;
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
         ddist = abs(*i1 - *i2);
         if (ddist > *i3/2) ddist = *i3 - ddist;
         distance += ddist*ddist;
      }
      distance=sqrt(distance);
   }

   float weight = _max*exp(-(distance*distance)/(2.0*_sigma*_sigma));

   NDPairList ndpl;

   std::string name="weight";
   FloatDataItem* fdi=new FloatDataItem(weight);
   std::auto_ptr<DataItem> fdi_ap(fdi);
   NDPair* ndp = new NDPair(name, fdi_ap);
   ndpl.push_back(ndp);

   name="distance";
   FloatDataItem* fdi2=new FloatDataItem(distance);
   std::auto_ptr<DataItem> fdi_ap2(fdi2);
   NDPair* ndp2 = new NDPair(name, fdi_ap2);
   ndpl.push_back(ndp2);

   std::auto_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(pset);
   pset->set(ndpl);
   return pset;
}

ReversedSrcRefGaussianWeightModifier::ReversedSrcRefGaussianWeightModifier() 
   : CG_ReversedSrcRefGaussianWeightModifierBase()
{
}

ReversedSrcRefGaussianWeightModifier::~ReversedSrcRefGaussianWeightModifier() 
{
}

void ReversedSrcRefGaussianWeightModifier::duplicate(std::auto_ptr<ReversedSrcRefGaussianWeightModifier>& dup) const
{
   dup.reset(new ReversedSrcRefGaussianWeightModifier(*this));
}

void ReversedSrcRefGaussianWeightModifier::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ReversedSrcRefGaussianWeightModifier(*this));
}

void ReversedSrcRefGaussianWeightModifier::duplicate(std::auto_ptr<CG_ReversedSrcRefGaussianWeightModifierBase>& dup) const
{
   dup.reset(new ReversedSrcRefGaussianWeightModifier(*this));
}

