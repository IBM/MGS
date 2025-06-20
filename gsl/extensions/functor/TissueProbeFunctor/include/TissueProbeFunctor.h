// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TissueProbeFunctor_H
#define TissueProbeFunctor_H

#include "Mgs.h"
#include "CG_TissueProbeFunctorBase.h"
#include "GslContext.h"
#include "NodeSet.h"
#include "TissueElement.h"
#include "NodeDescriptor.h"
#include "Grid.h"
#include <memory>

class Grid;

class TissueProbeFunctor : public CG_TissueProbeFunctorBase, public TissueElement
{
   public:
      void userInitialize(GslContext* CG_c);
      std::unique_ptr<NodeSet> userExecute(GslContext* CG_c);
      TissueProbeFunctor();
      TissueProbeFunctor(TissueProbeFunctor*);
      virtual ~TissueProbeFunctor();
      virtual void duplicate(std::unique_ptr<TissueProbeFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TissueProbeFunctorBase>&& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
      Grid* _grid;
      std::vector<NodeDescriptor*> _nodeDescriptors;
};

#endif

