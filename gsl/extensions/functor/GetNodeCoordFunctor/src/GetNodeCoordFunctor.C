// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GetNodeCoordFunctor.h"
#include "CG_GetNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "GenericService.h"
#include "CoordsStruct.h"

void GetNodeCoordFunctor::userInitialize(LensContext* CG_c) 
{
}

Service* GetNodeCoordFunctor::userExecute(LensContext* CG_c, Node*& node) 
{
   delete _service;
// General way if the node needs to return  ShallowArray<unsigned>
//    ShallowArray<unsigned> coords;
//    node->getNodeCoords(coords);
//    ShallowArray<unsigned>::iterator it, end = coords.end();
//    for (it = coords.begin(); it != end; ++it) {
//       _coords.coords.push_back(*it);
//    }
   node->getNodeCoords(_coords.coords);
   // Careful, publishable is 0, instead of having 
   // a Publishable, publisher etc,
   _service = new GenericService<CoordsStruct>(this, &_coords);
   return _service;
}

GetNodeCoordFunctor::GetNodeCoordFunctor() 
   : CG_GetNodeCoordFunctorBase(), _service(0)
{
}

GetNodeCoordFunctor::~GetNodeCoordFunctor() 
{
   destructOwnedHeap();
}

GetNodeCoordFunctor::GetNodeCoordFunctor(
   const GetNodeCoordFunctor& rv)
   : CG_GetNodeCoordFunctorBase(rv), _service(0), _coords(rv._coords)
{
   copyOwnedHeap(rv);
}

GetNodeCoordFunctor& GetNodeCoordFunctor::operator=(
   const GetNodeCoordFunctor& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _coords = rv._coords;
   }
   return *this;
}

void GetNodeCoordFunctor::duplicate(std::unique_ptr<GetNodeCoordFunctor>&& dup) const
{
   dup.reset(new GetNodeCoordFunctor(*this));
}

void GetNodeCoordFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GetNodeCoordFunctor(*this));
}

void GetNodeCoordFunctor::duplicate(std::unique_ptr<CG_GetNodeCoordFunctorBase>&& dup) const
{
   dup.reset(new GetNodeCoordFunctor(*this));
}

void GetNodeCoordFunctor::copyOwnedHeap(const GetNodeCoordFunctor& rv)
{
   if (rv._service) {
      std::unique_ptr<Service> dup;
      rv._service->duplicate(dup);
      _service = dup.release();
   } else {
      _service = 0;
   }
}

void GetNodeCoordFunctor::destructOwnedHeap()
{
   delete _service;
}

const char* GetNodeCoordFunctor::getServiceName(void* data) const
{
   if (data == &(_coords)) {
      return "_coords";
   }
   return "Error in Service Name!";
}

const char* GetNodeCoordFunctor::getServiceDescription(void* data) const
{
   if (data == &(_coords)) {
      return "Service from GetNodeCoordFunctor";
   }
   return "Error in Service Description!";
}
