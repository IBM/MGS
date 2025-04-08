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

#ifndef GetDstNodeCoordFunctor_H
#define GetDstNodeCoordFunctor_H
#include "Lens.h"

#include "CG_GetDstNodeCoordFunctorBase.h"
#include "LensContext.h"
#include <memory>

#include "CoordsStruct.h"

class GetDstNodeCoordFunctor : public CG_GetDstNodeCoordFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c, int& dim);
      int userExecute(LensContext* CG_c);
      GetDstNodeCoordFunctor();
      virtual ~GetDstNodeCoordFunctor();
      virtual void duplicate(std::unique_ptr<GetDstNodeCoordFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetDstNodeCoordFunctorBase>&& dup) const;

   private:
      CoordsStruct _coords;
      int _dim;
};

#endif
