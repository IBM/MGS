// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<GetPostNodeCoordFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_GetPostNodeCoordFunctorBase>& dup) const;
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
