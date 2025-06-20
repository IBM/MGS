// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "TissueMGSifyFunctor.h"
#include "CG_TissueMGSifyFunctorBase.h"
#include "GslContext.h"
#include "TissueFunctor.h"
#include <memory>

void TissueMGSifyFunctor::userInitialize(GslContext* CG_c) 
{
}

void TissueMGSifyFunctor::userExecute(GslContext* CG_c) 
{
  _tissueFunctor->doMGSify(CG_c);
}

TissueMGSifyFunctor::TissueMGSifyFunctor() 
   : CG_TissueMGSifyFunctorBase()
{
}

TissueMGSifyFunctor::~TissueMGSifyFunctor() 
{
}

void TissueMGSifyFunctor::duplicate(std::unique_ptr<TissueMGSifyFunctor>&& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

void TissueMGSifyFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

void TissueMGSifyFunctor::duplicate(std::unique_ptr<CG_TissueMGSifyFunctorBase>&& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

