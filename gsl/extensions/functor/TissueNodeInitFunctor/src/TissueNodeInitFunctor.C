// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "TissueNodeInitFunctor.h"
#include "CG_TissueNodeInitFunctorBase.h"
#include "LensContext.h"
#include "TissueFunctor.h"
#include <memory>

void TissueNodeInitFunctor::userInitialize(LensContext* CG_c) 
{
}

void TissueNodeInitFunctor::userExecute(LensContext* CG_c) 
{
  _tissueFunctor->doNodeInit(CG_c);
}

TissueNodeInitFunctor::TissueNodeInitFunctor() 
   : CG_TissueNodeInitFunctorBase()
{
}

TissueNodeInitFunctor::TissueNodeInitFunctor(TissueNodeInitFunctor const& tnif) 
  : CG_TissueNodeInitFunctorBase(), _tissueFunctor(tnif._tissueFunctor)
{
}

TissueNodeInitFunctor::~TissueNodeInitFunctor() 
{
}

void TissueNodeInitFunctor::duplicate(std::unique_ptr<TissueNodeInitFunctor>&& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

void TissueNodeInitFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

void TissueNodeInitFunctor::duplicate(std::unique_ptr<CG_TissueNodeInitFunctorBase>&& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

