// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "Neg.h"
#include "CG_NegBase.h"
#include "LensContext.h"
#include <memory>

void Neg::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

double Neg::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Neg, second argument: functor did not return a number");
  }
  return -ndi->getDouble();
}

Neg::Neg() 
   : CG_NegBase()
{
}

Neg::~Neg() 
{
}

void Neg::duplicate(std::unique_ptr<Neg>& dup) const
{
   dup.reset(new Neg(*this));
}

void Neg::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new Neg(*this));
}

void Neg::duplicate(std::unique_ptr<CG_NegBase>& dup) const
{
   dup.reset(new Neg(*this));
}

