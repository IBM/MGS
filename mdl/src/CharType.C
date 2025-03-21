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

#include "CharType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void CharType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new CharType(*this));
}

std::string CharType::getDescriptor() const
{
   return "char";
}

std::string CharType::getCapitalDescriptor() const
{
   return "Char";
}

CharType::~CharType() 
{
}
