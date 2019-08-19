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
#include "Scale.h"
#include "CG_ScaleBase.h"
#include "LensContext.h"
#include <memory>

void Scale::userInitialize(LensContext* CG_c, Functor*& f, double& scale) 
{
}

double Scale::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Scale, second argument: functor did not return a number");
  }
  return ndi->getDouble()*init.scale;  
}

Scale::Scale() 
   : CG_ScaleBase()
{
}

Scale::~Scale() 
{
}

void Scale::duplicate(std::unique_ptr<Scale>& dup) const
{
   dup.reset(new Scale(*this));
}

void Scale::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new Scale(*this));
}

void Scale::duplicate(std::unique_ptr<CG_ScaleBase>& dup) const
{
   dup.reset(new Scale(*this));
}

