// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void GetDstNodeCoordFunctor::duplicate(std::unique_ptr<GetDstNodeCoordFunctor>&& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}

void GetDstNodeCoordFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}

void GetDstNodeCoordFunctor::duplicate(std::unique_ptr<CG_GetDstNodeCoordFunctorBase>&& dup) const
{
   dup.reset(new GetDstNodeCoordFunctor(*this));
}
