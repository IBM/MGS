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

#ifndef GetPreNodeCoordFunctor_H
#define GetPreNodeCoordFunctor_H
//#include "Lens.h"

#include "CG_GetPreNodeCoordFunctorBase.h"
#include "LensContext.h"
#include "Service.h"
#include <memory>

#include "CoordsStruct.h"

//class CG_GetPreNodeCoordFunctorBase;
//class CoordsStruct;

class GetPreNodeCoordFunctor : public CG_GetPreNodeCoordFunctorBase, 
			       public Publishable
{
   public:
      void userInitialize(LensContext* CG_c);
      Service* userExecute(LensContext* CG_c, Edge*& edge);
      GetPreNodeCoordFunctor();
      virtual ~GetPreNodeCoordFunctor();
      GetPreNodeCoordFunctor(const GetPreNodeCoordFunctor& rv);
      GetPreNodeCoordFunctor& operator=(const GetPreNodeCoordFunctor& rv);
      virtual void duplicate(std::auto_ptr<GetPreNodeCoordFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_GetPreNodeCoordFunctorBase>& dup) const;
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
