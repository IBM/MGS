// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GetPostNodeCoordFunctor_H
#define GetPostNodeCoordFunctor_H
#include "Lens.h"

#include "CG_GetPostNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "CoordsStruct.h"

class GetPostNodeCoordFunctor : public CG_GetPostNodeCoordFunctorBase,
				public Publishable
{
   public:
      void userInitialize(LensContext* CG_c);
      Service* userExecute(LensContext* CG_c, Edge*& edge);
      GetPostNodeCoordFunctor();
      GetPostNodeCoordFunctor(const GetPostNodeCoordFunctor& rv);
      GetPostNodeCoordFunctor& operator=(const GetPostNodeCoordFunctor& rv);
      virtual ~GetPostNodeCoordFunctor();
      virtual void duplicate(std::unique_ptr<GetPostNodeCoordFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetPostNodeCoordFunctorBase>&& dup) const;
      virtual Publisher* getPublisher() {
	 return 0;
      }
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
   private:
      void copyOwnedHeap(const GetPostNodeCoordFunctor& rv);
      void destructOwnedHeap();
      Service* _service;
      CoordsStruct _coords;
};

#endif
