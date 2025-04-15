// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TissueLayoutFunctor_H
#define TissueLayoutFunctor_H

#include "Lens.h"
#include "CG_TissueLayoutFunctorBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "TissueElement.h"
#include <memory>

class TissueFunctor;

class TissueLayoutFunctor : public CG_TissueLayoutFunctorBase, public TissueElement
{
   public:
      void userInitialize(LensContext* CG_c);
      ShallowArray< int > userExecute(LensContext* CG_c);
      TissueLayoutFunctor();
      TissueLayoutFunctor(TissueLayoutFunctor const &);
      virtual ~TissueLayoutFunctor();
      virtual void duplicate(std::unique_ptr<TissueLayoutFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TissueLayoutFunctorBase>&& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
