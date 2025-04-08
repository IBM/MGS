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

#include "RefAngleModifier.h"
#include "CG_RefAngleModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>


#include "NDPair.h"
#include "NDPairList.h"
#include "FloatDataItem.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <cmath>
#include "LensContext.h"
#include "ParameterSet.h"
#include <vector>
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include <math.h>


std::unique_ptr<ParameterSet> RefAngleModifier::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "DstRefAngleModifier: functor did not return a Parameter Set");
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

   float angle =M_PI/2.0;
   std::vector<int> coords1, coords2, gridSize;
   n1->getNodeCoords(coords1);
   n2->getNodeCoords(coords2);
   float x1, y1;
   float xRef, yRef;
   x1 = coords1[0];
   y1 = coords1[1];
   xRef = coords2[0];
   yRef = coords2[1];
   float xdiff, ydiff;

   xdiff = x1-xRef;
   ydiff = y1-yRef; 

   if (init.wrapFlag ==1){
      Grid *g = n2->getGridLayerDescriptor()->getGrid();
      gridSize = g->getSize();
  
      if (xdiff >   gridSize[0]/2) xdiff -= gridSize[0];
      if (xdiff <= -gridSize[0]/2) xdiff += gridSize[0];
      if (ydiff >   gridSize[1]/2) ydiff -= gridSize[1];
      if (ydiff <= -gridSize[1]/2) ydiff += gridSize[1];
   }
 

   if (ydiff ==0 && xdiff==0) angle = 0;
   else angle = atan2(ydiff,xdiff);
  

 
   std::string name="angle";
   FloatDataItem* fdi=new FloatDataItem(angle);
   std::unique_ptr<DataItem> fdi_ap(fdi);

   NDPair* ndp = new NDPair(name, fdi_ap);
   NDPairList ndpl;
   ndpl.push_back(ndp);
   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(std::move(pset));
   pset->set(ndpl);
   return pset;
}

void RefAngleModifier::userInitialize(LensContext* CG_c, int& directionFlag, int& wrapFlag, Functor*& f) 
{
}

RefAngleModifier::RefAngleModifier() 
   : CG_RefAngleModifierBase()
{
}

RefAngleModifier::~RefAngleModifier() 
{
}

void RefAngleModifier::duplicate(std::unique_ptr<RefAngleModifier>&& dup) const
{
   dup.reset(new RefAngleModifier(*this));
}

void RefAngleModifier::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new RefAngleModifier(*this));
}

void RefAngleModifier::duplicate(std::unique_ptr<CG_RefAngleModifierBase>&& dup) const
{
   dup.reset(new RefAngleModifier(*this));
}

