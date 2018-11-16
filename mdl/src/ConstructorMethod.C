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

#include "ConstructorMethod.h"
#include "Method.h"
#include "Attribute.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>
#include <cassert>

ConstructorMethod::ConstructorMethod()
   : Method(), _initializationStr("")
{
}

ConstructorMethod::ConstructorMethod(const std::string& name, 
				     const std::string& returnStr,
				     const std::string& functionBody,
				     const std::string& initializationStr) 
   : Method(name, returnStr, functionBody)
   , _initializationStr(initializationStr)
{
}

void ConstructorMethod::duplicate(std::auto_ptr<Method>& dup) const
{
   dup.reset(new ConstructorMethod(*this));
}

ConstructorMethod::~ConstructorMethod()
{
}

const std::string& ConstructorMethod::getInitializationStr() const
{
   return _initializationStr;
}

void ConstructorMethod::setInitializationStr(
   const std::string& initializationStr)
{
   _initializationStr = initializationStr;
}

void ConstructorMethod::internalAddConstructorInitializer(
   const std::vector<Attribute*>& attributes
   , const std::string& beginning
   , const std::string& copyFrom) 
{
   std::string inits = "";
   bool first = true;
   if (attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = attributes.end();
      std::string eachInit;
      for (it = attributes.begin(); it != end; it++) {
	 callInitMethod(it, eachInit, copyFrom);
	 if (eachInit != "") {
	    if (first) {
	       first = false;
	    } else {
	       /* if the attribute is wrapped inside the macro-check, then the comma is added already*/
	       if ((*it)->getMacroConditional().getName() == "")
		  inits += ", ";
	    }
	    inits += eachInit;
	 }
      }
   }
   if (inits == "") {
      _initializationStr = beginning;
   } else {
      if (beginning != "") {	 
	 _initializationStr = beginning + ", " + inits;
      } else {
	 _initializationStr = inits;
      }
   }
}

std::string ConstructorMethod::printConstructorExtra()
{
   std::string retVal;
   if (_initializationStr != "") {
      retVal = TAB;
      if (isInline()) retVal += TAB;
      retVal += ": " + _initializationStr + "\n";
   }
   return retVal;
}

void ConstructorMethod::callInitMethod(
   const std::vector<Attribute*>::const_iterator& it,
   std::string& initStr, const std::string& copyFrom)
{
   // this function has to be overriden in child classes that use
   // internalAddConstructorInitializer, since this function is a hook method.
   assert(0);
}
