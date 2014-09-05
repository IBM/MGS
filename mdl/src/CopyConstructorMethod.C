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

void CopyConstructorMethod::duplicate(std::auto_ptr<Method>& dup) const
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
