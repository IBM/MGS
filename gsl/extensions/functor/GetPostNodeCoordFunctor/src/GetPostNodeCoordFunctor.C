// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GetPostNodeCoordFunctor.h"
#include "CG_GetPostNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "NodeDescriptor.h"
#include "GenericService.h"
#include "CoordsStruct.h"

void GetPostNodeCoordFunctor::userInitialize(LensContext* CG_c) 
{
}

Service* GetPostNodeCoordFunctor::userExecute(LensContext* CG_c, Edge*& edge) 
{
   delete _service;
   NodeDescriptor* node = edge->getPostNode();

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

GetPostNodeCoordFunctor::GetPostNodeCoordFunctor() 
   : CG_GetPostNodeCoordFunctorBase(), _service(0)
{
}

GetPostNodeCoordFunctor::~GetPostNodeCoordFunctor() 
{
   destructOwnedHeap();
}

GetPostNodeCoordFunctor::GetPostNodeCoordFunctor(
   const GetPostNodeCoordFunctor& rv)
   : CG_GetPostNodeCoordFunctorBase(rv), _service(0), _coords(rv._coords)
{
   copyOwnedHeap(rv);
}

GetPostNodeCoordFunctor& GetPostNodeCoordFunctor::operator=(
   const GetPostNodeCoordFunctor& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _coords = rv._coords;
   }
   return *this;
}

void GetPostNodeCoordFunctor::duplicate(std::unique_ptr<GetPostNodeCoordFunctor>&& dup) const
{
   dup.reset(new GetPostNodeCoordFunctor(*this));
}

void GetPostNodeCoordFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GetPostNodeCoordFunctor(*this));
}

void GetPostNodeCoordFunctor::duplicate(std::unique_ptr<CG_GetPostNodeCoordFunctorBase>&& dup) const
{
   dup.reset(new GetPostNodeCoordFunctor(*this));
}

void GetPostNodeCoordFunctor::copyOwnedHeap(const GetPostNodeCoordFunctor& rv)
{
   if (rv._service) {
      std::unique_ptr<Service> dup;
      rv._service->duplicate(dup);
      _service = dup.release();
   } else {
      _service = 0;
   }
}

void GetPostNodeCoordFunctor::destructOwnedHeap()
{
   delete _service;
}

const char* GetPostNodeCoordFunctor::getServiceName(void* data) const
{
   if (data == &(_coords)) {
      return "_coords";
   }
   return "Error in Service Name!";
}

const char* GetPostNodeCoordFunctor::getServiceDescription(void* data) const
{
   if (data == &(_coords)) {
      return "Service from GetPostNodeCoordFunctor";
   }
   return "Error in Service Description!";
}
