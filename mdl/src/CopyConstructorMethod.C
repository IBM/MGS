#include <memory>
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

#include "CopyConstructorMethod.h"
#include "Method.h"
#include "Attribute.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>

CopyConstructorMethod::CopyConstructorMethod()
   : ConstructorMethod()
{
}

CopyConstructorMethod::CopyConstructorMethod(
   const std::string& name, 
   const std::string& returnStr,
   const std::string& functionBody,
   const std::string& initializationStr) 
   : ConstructorMethod(name, returnStr, functionBody, initializationStr)
{
}

void CopyConstructorMethod::duplicate(std::unique_ptr<Method>&& dup) const
{
   dup.reset(new CopyConstructorMethod(*this));
}

CopyConstructorMethod::~CopyConstructorMethod()
{
}

void CopyConstructorMethod::addCopyConstructorInitializers(
   const std::vector<Attribute*>& attributes
   , const std::string& beginning
   , const std::string& copyFrom) 
{
   internalAddConstructorInitializer(attributes, beginning, copyFrom);
}

void CopyConstructorMethod::callInitMethod(
   const std::vector<Attribute*>::const_iterator& it,
   std::string& initStr, const std::string& copyFrom)
{
   (*it)->fillCopyInitializer(initStr, copyFrom);
}
