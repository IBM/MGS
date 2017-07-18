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

#include "DefaultConstructorMethod.h"
#include "Method.h"
#include "Attribute.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>

DefaultConstructorMethod::DefaultConstructorMethod()
   : ConstructorMethod()
{
}

DefaultConstructorMethod::DefaultConstructorMethod(
   const std::string& name
   , const std::string& returnStr
   , const std::string& functionBody
   , const std::string& initializationStr) 
   : ConstructorMethod(name, returnStr, functionBody, initializationStr)
{
}

void DefaultConstructorMethod::duplicate(std::auto_ptr<Method>& dup) const
{
   dup.reset(new DefaultConstructorMethod(*this));
}


DefaultConstructorMethod::~DefaultConstructorMethod()
{
}

void DefaultConstructorMethod::addDefaultConstructorParameters(
   const std::vector<Attribute*>& attributes,
   const std::string& className) 
{
   if (attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = attributes.end();
      std::string parameter;
      for (it = attributes.begin(); it != end; it++) {
	 parameter = (*it)->getConstructorParameter(className);
	 if (parameter != "") {
	    addParameter(parameter);
	 }
      }
   }
}

void DefaultConstructorMethod::addDefaultConstructorInitializers(
   const std::vector<Attribute*>& attributes,
   const std::string& beginning) 
{
   internalAddConstructorInitializer(attributes, beginning);
}

void DefaultConstructorMethod::callInitMethod(
   const std::vector<Attribute*>::const_iterator& it,
   std::string& initStr, const std::string& copyFrom)
{
   (*it)->fillInitializer(initStr);
}
