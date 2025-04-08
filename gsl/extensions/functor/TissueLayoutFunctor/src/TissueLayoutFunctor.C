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

