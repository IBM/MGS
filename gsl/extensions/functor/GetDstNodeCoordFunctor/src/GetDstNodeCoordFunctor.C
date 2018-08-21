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

#include "GetDstNodeCoordFunctor.h"
#include "CG_GetDstNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "GenericService.h"
#include "CoordsStruct.h"
#include "ConnectionContext.h"
#include "NodeDescriptor.h"

void GetDstNodeCoordFunctor::userInitialize(LensContext* CG_c, int& dim) 
{
  _dim=dim;
}

int GetDstNodeCoordFunctor::userExecute(LensContext* CG_c)
{
   CG_c->connectionContext->destinationNode->getNodeCoords(_coords.coords);
   return _coords.coords[_dim];
}

GetDstNodeCoordFunctor::GetDstNodeCoordFunctor() 
   : CG_GetDstNodeCoordFunctorBase()
{
}

GetDstNodeCoordFunctor::~GetDstNodeCoordFunctor() 
{
}

void GetDstNodeCoordFunctor::duplicate(std::auto_ptr<GetDstNodeCoordFunctor>& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}

void GetDstNodeCoordFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}

void GetDstNodeCoordFunctor::duplicate(std::auto_ptr<CG_GetDstNodeCoordFunctorBase>& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}
