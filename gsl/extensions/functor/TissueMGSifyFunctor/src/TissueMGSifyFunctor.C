// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "TissueMGSifyFunctor.h"
#include "CG_TissueMGSifyFunctorBase.h"
#include "LensContext.h"
#include "TissueFunctor.h"
#include <memory>

void TissueMGSifyFunctor::userInitialize(LensContext* CG_c) 
{
}

void TissueMGSifyFunctor::userExecute(LensContext* CG_c) 
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

void TissueMGSifyFunctor::duplicate(std::unique_ptr<TissueMGSifyFunctor>& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

void TissueMGSifyFunctor::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

void TissueMGSifyFunctor::duplicate(std::unique_ptr<CG_TissueMGSifyFunctorBase>& dup) const
{
   dup.reset(new TissueMGSifyFunctor(*this));
}

