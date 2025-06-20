// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TissueMGSifyFunctor_H
#define TissueMGSifyFunctor_H

#include "Mgs.h"
#include "CG_TissueMGSifyFunctorBase.h"
#include "GslContext.h"
#include "TissueElement.h"
#include <memory>

class TissueMGSifyFunctor : public CG_TissueMGSifyFunctorBase, public TissueElement
{
   public:
      void userInitialize(GslContext* CG_c);
      void userExecute(GslContext* CG_c);
      TissueMGSifyFunctor();
      virtual ~TissueMGSifyFunctor();
      virtual void duplicate(std::unique_ptr<TissueMGSifyFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TissueMGSifyFunctorBase>&& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
