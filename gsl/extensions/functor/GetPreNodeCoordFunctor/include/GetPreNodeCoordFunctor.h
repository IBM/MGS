// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GetPreNodeCoordFunctor_H
#define GetPreNodeCoordFunctor_H
//#include "Mgs.h"

#include "CG_GetPreNodeCoordFunctorBase.h"
#include "GslContext.h"
#include "Service.h"
#include <memory>

#include "CoordsStruct.h"

//class CG_GetPreNodeCoordFunctorBase;
//class CoordsStruct;

class GetPreNodeCoordFunctor : public CG_GetPreNodeCoordFunctorBase, 
			       public Publishable
{
   public:
      void userInitialize(GslContext* CG_c);
      Service* userExecute(GslContext* CG_c, Edge*& edge);
      GetPreNodeCoordFunctor();
      virtual ~GetPreNodeCoordFunctor();
      GetPreNodeCoordFunctor(const GetPreNodeCoordFunctor& rv);
      GetPreNodeCoordFunctor& operator=(const GetPreNodeCoordFunctor& rv);
      virtual void duplicate(std::unique_ptr<GetPreNodeCoordFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetPreNodeCoordFunctorBase>&& dup) const;
      virtual Publisher* getPublisher() {
	 return 0;
      }
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
   private:
      void copyOwnedHeap(const GetPreNodeCoordFunctor& rv);
      void destructOwnedHeap();
      Service* _service;
      CoordsStruct _coords;
};

#endif
