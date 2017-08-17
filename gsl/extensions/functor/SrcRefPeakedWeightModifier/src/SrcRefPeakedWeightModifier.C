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

#include "SrcRefPeakedWeightModifier.h"
#include "CG_SrcRefPeakedWeightModifierBase.h"
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

void SrcRefPeakedWeightModifier::userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance) 
{
  _sigma=sigma;
  _max=max;
  _wrapDistance=wrapDistance;
}

std::auto_ptr<ParameterSet> SrcRefPeakedWeightModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "SrcRefPeakedWeightModifier: functor did not return a Parameter Set");
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

   float weight = 1.0-pow((distance/_max), _sigma);
   if (weight<0) weight=0;

   NDPairList ndpl;

   std::string name="weight";
   FloatDataItem* fdi=new FloatDataItem(weight);
   std::auto_ptr<DataItem> fdi_ap(fdi);
   NDPair* ndp = new NDPair(name, fdi_ap);
   ndpl.push_back(ndp);

   name="distance";
   FloatDataItem* fdi2=new FloatDataItem(distance/_max);
   std::auto_ptr<DataItem> fdi_ap2(fdi2);
   NDPair* ndp2 = new NDPair(name, fdi_ap2);
   ndpl.push_back(ndp2);

   std::auto_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(pset);
   pset->set(ndpl);
   return pset;
}

SrcRefPeakedWeightModifier::SrcRefPeakedWeightModifier() 
   : CG_SrcRefPeakedWeightModifierBase()
{
}

SrcRefPeakedWeightModifier::~SrcRefPeakedWeightModifier() 
{
}

void SrcRefPeakedWeightModifier::duplicate(std::auto_ptr<SrcRefPeakedWeightModifier>& dup) const
{
   dup.reset(new SrcRefPeakedWeightModifier(*this));
}

void SrcRefPeakedWeightModifier::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new SrcRefPeakedWeightModifier(*this));
}

void SrcRefPeakedWeightModifier::duplicate(std::auto_ptr<CG_SrcRefPeakedWeightModifierBase>& dup) const
{
   dup.reset(new SrcRefPeakedWeightModifier(*this));
}

