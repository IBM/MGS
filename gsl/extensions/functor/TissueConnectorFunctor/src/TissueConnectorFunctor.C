// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "TissueConnectorFunctor.h"
#include "CG_TissueConnectorFunctorBase.h"
#include "GslContext.h"
#include "TissueFunctor.h"
#include <memory>

void TissueConnectorFunctor::userInitialize(GslContext* CG_c) 
{
}

void TissueConnectorFunctor::userExecute(GslContext* CG_c) 
{
  _tissueFunctor->doConnector(CG_c);
}

TissueConnectorFunctor::TissueConnectorFunctor() 
   : CG_TissueConnectorFunctorBase()
{
}

TissueConnectorFunctor::TissueConnectorFunctor(TissueConnectorFunctor const& tnif) 
  : CG_TissueConnectorFunctorBase(), _tissueFunctor(tnif._tissueFunctor)
{
}

TissueConnectorFunctor::~TissueConnectorFunctor() 
{
}

void TissueConnectorFunctor::duplicate(std::unique_ptr<TissueConnectorFunctor>&& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

void TissueConnectorFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

void TissueConnectorFunctor::duplicate(std::unique_ptr<CG_TissueConnectorFunctorBase>&& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

