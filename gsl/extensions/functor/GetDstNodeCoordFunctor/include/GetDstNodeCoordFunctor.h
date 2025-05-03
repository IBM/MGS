// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GetDstNodeCoordFunctor_H
#define GetDstNodeCoordFunctor_H
#include "Mgs.h"

#include "CG_GetDstNodeCoordFunctorBase.h"
#include "GslContext.h"
#include <memory>

#include "CoordsStruct.h"

class GetDstNodeCoordFunctor : public CG_GetDstNodeCoordFunctorBase
{
   public:
      void userInitialize(GslContext* CG_c, int& dim);
      int userExecute(GslContext* CG_c);
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
