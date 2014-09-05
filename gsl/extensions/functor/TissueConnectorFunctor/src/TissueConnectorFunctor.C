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
#include "TissueConnectorFunctor.h"
#include "CG_TissueConnectorFunctorBase.h"
#include "LensContext.h"
#include "TissueFunctor.h"
#include <memory>

void TissueConnectorFunctor::userInitialize(LensContext* CG_c) 
{
}

void TissueConnectorFunctor::userExecute(LensContext* CG_c) 
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

void TissueConnectorFunctor::duplicate(std::auto_ptr<TissueConnectorFunctor>& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

void TissueConnectorFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

void TissueConnectorFunctor::duplicate(std::auto_ptr<CG_TissueConnectorFunctorBase>& dup) const
{
   dup.reset(new TissueConnectorFunctor(*this));
}

