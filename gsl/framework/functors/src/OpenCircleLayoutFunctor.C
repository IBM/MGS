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

#include "OpenCircleLayoutFunctor.h"
#include "FunctorType.h"
#include "NumericDataItem.h"
#include "LensContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "Grid.h"
#include "LayerDefinitionContext.h"
#include "SyntaxErrorException.h"

#include <math.h>
//#include <iostream>
#include <sstream>

void OpenCircleLayoutFunctor::doInitialize(LensContext *c, 
					   const std::vector<DataItem*>& args)
{
   NumericDataItem* nbrPositionsDI = 
      dynamic_cast<NumericDataItem*>(*(args.begin()));
   if (nbrPositionsDI == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in OpenCircleLayoutFunctor");
   }
   _nbrPositions=nbrPositionsDI->getInt();
}


void OpenCircleLayoutFunctor::doExecute(LensContext *c, 
					const std::vector<DataItem*>& args, 
					std::auto_ptr<DataItem>& rvalue)
{
   double radiansPerPosition = (2.0*M_PI)/double(_nbrPositions);
   Grid* g = c->layerContext->grid;
   std::vector<int> size = g->getSize();
   int nbrGridPts = g->getNbrGridNodes();

   int sz = size.size();
   if (sz!=2) {
      std::ostringstream msg;
      msg << "Only 2-D grids supported on OpenCircleLayoutFunctor" << std::endl
	  << sz << "-D grid specified." << std::endl;
	 throw SyntaxErrorException(msg.str());
   }
   int radius = size[0];
   for (int i = 1; i<sz; ++i) {
      int e = int(float(size[i]-1)/2.0);
      if (e<radius) radius = e;
   }
   float sn2 = sin(radiansPerPosition)*sin(radiansPerPosition);
   float cs2 = (cos(radiansPerPosition)-1)*(cos(radiansPerPosition)-1);
   if (radius<(sqrt(2.0)/sqrt(sn2+cs2))) {
      std::ostringstream msg;
      msg << "Specified grid is not large enough to accommodate circle with "
	  << _nbrPositions << " positions " << std::endl
	  << "For " << _nbrPositions 
	  << " positions, smallest grid dimension must be "
	  << (ceil((sqrt(2.0)/sqrt(sn2+cs2)))*2) << "." << std::endl;
	 throw SyntaxErrorException(msg.str());
   }

   std::vector<int> densityVec;
   for (int i = 0; i<nbrGridPts; ++i) densityVec.push_back(0);
   for (int i = 0; i<_nbrPositions; ++i) {
      float theta = i*radiansPerPosition;
      int y = (int(radius*(1+sin(theta))));
      int x = (int(radius*(1+cos(theta))));
      densityVec[(y*size[1])+x]=1;
      if (y<0 || y>size[0]-1 || x<0 || x>size[1]-1) {
	 std::ostringstream msg;
         msg << "X : " << x << " Y : " << y 
	     << " out of range in OpenCircleLayoutFunctor!" << std::endl;
	 throw SyntaxErrorException(msg.str());
      }
   }
   densityVec[(radius*size[1])+radius]=1;

   int sum=0;
   std::vector<int>::iterator iter = densityVec.begin();
   std::vector<int>::iterator end = densityVec.end();
   for (;iter!=end;++iter) {
      if (*iter !=0) ++sum;
   }
   if (sum!=_nbrPositions+1) {
      std::ostringstream msg;
      msg << "Unexpected density vector generated in OpenCircleLayoutFunctor. Number non-zero densities = " 
	  << sum << "!" << std::endl;
      throw SyntaxErrorException(msg.str());
   }

   std::vector<int> &dv = *_density.getModifiableIntVector();
   dv=densityVec;

   rvalue.reset(new IntArrayDataItem(_density));
}


void OpenCircleLayoutFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new OpenCircleLayoutFunctor(*this));
}


OpenCircleLayoutFunctor::OpenCircleLayoutFunctor()
: _nbrPositions(0)
{
}

OpenCircleLayoutFunctor::~OpenCircleLayoutFunctor()
{
}
