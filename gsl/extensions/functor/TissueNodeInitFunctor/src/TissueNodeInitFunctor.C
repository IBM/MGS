// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

void TissueNodeInitFunctor::duplicate(std::auto_ptr<TissueNodeInitFunctor>& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

void TissueNodeInitFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

void TissueNodeInitFunctor::duplicate(std::auto_ptr<CG_TissueNodeInitFunctorBase>& dup) const
{
   dup.reset(new TissueNodeInitFunctor(*this));
}

