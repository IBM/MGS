// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TissueConnectorFunctor_H
#define TissueConnectorFunctor_H

#include "Mgs.h"
#include "CG_TissueConnectorFunctorBase.h"
#include "GslContext.h"
#include "TissueElement.h"
#include <memory>

class TissueConnectorFunctor : public CG_TissueConnectorFunctorBase, public TissueElement
{
   public:
      void userInitialize(GslContext* CG_c);
      void userExecute(GslContext* CG_c);
      TissueConnectorFunctor();
      TissueConnectorFunctor(TissueConnectorFunctor const &);
      virtual ~TissueConnectorFunctor();
      virtual void duplicate(std::unique_ptr<TissueConnectorFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TissueConnectorFunctorBase>&& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
