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
#include "TissueProbeFunctor.h"
#include "CG_TissueProbeFunctorBase.h"
#include "LensContext.h"
#include "NodeSet.h"
#include "TissueFunctor.h"
#include <memory>

void TissueProbeFunctor::userInitialize(LensContext* CG_c) 
{
}

std::auto_ptr<NodeSet> TissueProbeFunctor::userExecute(LensContext* CG_c) 
{
  std::auto_ptr<NodeSet> rval;
  _tissueFunctor->doProbe(CG_c, rval);
  return rval;
}

TissueProbeFunctor::TissueProbeFunctor() 
   : CG_TissueProbeFunctorBase()
{
}

TissueProbeFunctor::~TissueProbeFunctor() 
{
}

void TissueProbeFunctor::duplicate(std::auto_ptr<TissueProbeFunctor>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

void TissueProbeFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

void TissueProbeFunctor::duplicate(std::auto_ptr<CG_TissueProbeFunctorBase>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

