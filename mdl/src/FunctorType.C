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

#include "FunctorType.h"
#include "LensType.h"
#include <string>
#include "DataType.h"
#include <memory>

void FunctorType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new FunctorType(*this));
}

std::string FunctorType::getDescriptor() const
{
   return "Functor";
}

bool FunctorType::shouldBeOwned() const
{
   return true;
}

FunctorType::~FunctorType() 
{
}
