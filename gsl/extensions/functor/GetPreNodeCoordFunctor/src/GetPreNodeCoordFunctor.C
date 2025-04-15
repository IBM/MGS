// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GetPreNodeCoordFunctor.h"
//#include "CG_GetPreNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "NodeDescriptor.h"
#include "GenericService.h"
//#include "CoordsStruct.h"

void GetPreNodeCoordFunctor::userInitialize(LensContext* CG_c) 
{
}

Service* GetPreNodeCoordFunctor::userExecute(LensContext* CG_c, Edge*& edge) 
{
   delete _service;
   NodeDescriptor* node = edge->getPreNode();

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

GetPreNodeCoordFunctor::GetPreNodeCoordFunctor() 
   : CG_GetPreNodeCoordFunctorBase(), _service(0)
{
}

GetPreNodeCoordFunctor::~GetPreNodeCoordFunctor() 
{
   destructOwnedHeap();
}

GetPreNodeCoordFunctor::GetPreNodeCoordFunctor(
   const GetPreNodeCoordFunctor& rv)
   : CG_GetPreNodeCoordFunctorBase(rv), _service(0), _coords(rv._coords)
{
   copyOwnedHeap(rv);
}

GetPreNodeCoordFunctor& GetPreNodeCoordFunctor::operator=(
   const GetPreNodeCoordFunctor& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _coords = rv._coords;
   }
   return *this;
}

void GetPreNodeCoordFunctor::duplicate(std::unique_ptr<GetPreNodeCoordFunctor>&& dup) const 
{
   dup.reset(new GetPreNodeCoordFunctor(*this));
}

void GetPreNodeCoordFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GetPreNodeCoordFunctor(*this));
}

void GetPreNodeCoordFunctor::duplicate(std::unique_ptr<CG_GetPreNodeCoordFunctorBase>&& dup) const
{
   dup.reset(new GetPreNodeCoordFunctor(*this));
}

void GetPreNodeCoordFunctor::copyOwnedHeap(const GetPreNodeCoordFunctor& rv)
{
   if (rv._service) {
      std::unique_ptr<Service> dup;
      rv._service->duplicate(dup);
      _service = dup.release();
   } else {
      _service = 0;
   }
}

void GetPreNodeCoordFunctor::destructOwnedHeap()
{
   delete _service;
}

const char* GetPreNodeCoordFunctor::getServiceName(void* data) const
{
   if (data == &(_coords)) {
      return "_coords";
   }
   return "Error in Service Name!";
}

const char* GetPreNodeCoordFunctor::getServiceDescription(void* data) const
{
   if (data == &(_coords)) {
      return "Service from GetPreNodeCoordFunctor";
   }
   return "Error in Service Description!";
}
