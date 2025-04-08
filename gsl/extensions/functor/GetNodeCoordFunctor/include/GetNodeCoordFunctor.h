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

#ifndef GetNodeCoordFunctor_H
#define GetNodeCoordFunctor_H
#include "Lens.h"

#include "CG_GetNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "CoordsStruct.h"

class GetNodeCoordFunctor : public CG_GetNodeCoordFunctorBase,
			    public Publishable
{
   public:
      void userInitialize(LensContext* CG_c);
      Service* userExecute(LensContext* CG_c, Node*& node);
      GetNodeCoordFunctor();
      GetNodeCoordFunctor(const GetNodeCoordFunctor& rv);
      GetNodeCoordFunctor& operator=(const GetNodeCoordFunctor& rv);
      virtual ~GetNodeCoordFunctor();
      virtual void duplicate(std::unique_ptr<GetNodeCoordFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetNodeCoordFunctorBase>&& dup) const;
      virtual Publisher* getPublisher() {
	 return 0;
      }
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
   private:
      void copyOwnedHeap(const GetNodeCoordFunctor& rv);
      void destructOwnedHeap();
      Service* _service;
      CoordsStruct _coords;
};

#endif
