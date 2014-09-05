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

#include "RefDistanceModifier.h"
#include "CG_RefDistanceModifierBase.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <cmath>
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>
#include <vector>
#include "Grid.h"
#include "GridLayerDescriptor.h"


void RefDistanceModifier::userInitialize(LensContext* CG_c, int& directionFlag, int& wrapFlag, Functor*& f) 
{
}

std::auto_ptr<ParameterSet> RefDistanceModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "DstRefDistanceModifier: functor did not return a Parameter Set");
   }

   NodeDescriptor* n1;
   NodeDescriptor* n2;

   if (init.directionFlag==0){
      n1 = cc->sourceNode;
      n2 = cc->sourceRefNode;
   }
   else{
      n1 = cc->destinationNode;
      n2 = cc->destinationRefNode;
   }

   assert(n1);
   assert(n2);

   float distance=0, ddist;
   std::vector<int> coords1, coords2, gridSize;
   n1->getNodeCoords(coords1);
   n2->getNodeCoords(coords2);
   std::vector<int>::iterator i1, i2, i3, end1 = coords1.end();

   if (init.wrapFlag ==0){
      for (i1=coords1.begin(),i2=coords2.begin();i1!=end1;++i1, ++i2){
         ddist = *i1 - *i2;
         distance += ddist*ddist;
      }
      distance=sqrt(distance);
   } 
   else{
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

RefDistanceModifier::RefDistanceModifier() 
   : CG_RefDistanceModifierBase()
{
}

RefDistanceModifier::~RefDistanceModifier() 
{
}

void RefDistanceModifier::duplicate(std::auto_ptr<RefDistanceModifier>& dup) const
{
   dup.reset(new RefDistanceModifier(*this));
}

void RefDistanceModifier::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new RefDistanceModifier(*this));
}

void RefDistanceModifier::duplicate(std::auto_ptr<CG_RefDistanceModifierBase>& dup) const
{
   dup.reset(new RefDistanceModifier(*this));
}

