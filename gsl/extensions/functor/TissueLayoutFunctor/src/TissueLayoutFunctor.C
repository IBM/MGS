// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "TissueLayoutFunctor.h"
#include "CG_TissueLayoutFunctorBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "TissueFunctor.h"
#include <memory>

void TissueLayoutFunctor::userInitialize(LensContext* CG_c) 
{
}

ShallowArray< int > TissueLayoutFunctor::userExecute(LensContext* CG_c) 
{
  return _tissueFunctor->doLayout(CG_c);
}

TissueLayoutFunctor::TissueLayoutFunctor() 
  : CG_TissueLayoutFunctorBase(), _tissueFunctor(0)
{
}

TissueLayoutFunctor::TissueLayoutFunctor(TissueLayoutFunctor const& tlf) 
  : CG_TissueLayoutFunctorBase(), _tissueFunctor(tlf._tissueFunctor)
{
}

TissueLayoutFunctor::~TissueLayoutFunctor() 
{
}

void TissueLayoutFunctor::duplicate(std::unique_ptr<TissueLayoutFunctor>&& dup) const
{
   dup.reset(new TissueLayoutFunctor(*this));
}

void TissueLayoutFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new TissueLayoutFunctor(*this));
}

void TissueLayoutFunctor::duplicate(std::unique_ptr<CG_TissueLayoutFunctorBase>&& dup) const
{
   dup.reset(new TissueLayoutFunctor(*this));
}

