// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TissueNodeInitFunctor_H
#define TissueNodeInitFunctor_H

#include "Lens.h"
#include "CG_TissueNodeInitFunctorBase.h"
#include "LensContext.h"
#include "TissueElement.h"
#include <memory>

class TissueNodeInitFunctor : public CG_TissueNodeInitFunctorBase, public TissueElement
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c);
      TissueNodeInitFunctor();
      TissueNodeInitFunctor(TissueNodeInitFunctor const &);
      virtual ~TissueNodeInitFunctor();
      virtual void duplicate(std::unique_ptr<TissueNodeInitFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TissueNodeInitFunctorBase>&& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
