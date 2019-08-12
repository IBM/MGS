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
#include <iostream>
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
   bool on_first_round = true;
   bool previous_is_inside_macro_check = false; /* tell if the previous param is inside a macro check or not [if YES, then a new line is already added (and the current param, regardless inside a macro-check or not, can be safely added); if NO, then we need to add a new line if the current param is inside a macro-check]
						   */
   bool current_is_inside_macro_check = true;
   if (attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = attributes.end();
      std::string eachInit;
      for (it = attributes.begin(); it != end; it++) {
	 callInitMethod(it, eachInit, copyFrom);
	 if (eachInit != "") {
	    if (first) {
	       first = false;
	    } else {
	       on_first_round = false;
	       /* if the attribute is wrapped inside the macro-check, then the comma is added already*/
	       if ((*it)->getMacroConditional().getName() == "")
	       {
		  current_is_inside_macro_check = false;
	       }
	       else
		  current_is_inside_macro_check = true;
	       if (previous_is_inside_macro_check)
		  inits += TAB + ", ";
	       else
		  inits += ", ";
	    }
	    if (previous_is_inside_macro_check)
	    {
	       inits += eachInit;
	    }else{
	       if (current_is_inside_macro_check == true)
	       {
		  if (on_first_round)
		     inits += eachInit;
		  else
		     inits += "\n" + eachInit;
	       }
	    }
	    previous_is_inside_macro_check = current_is_inside_macro_check;
	 }
      }
   }
   if (inits == "") {
      _initializationStr = beginning;
   } else {
      if (beginning != "") {
	if (inits.find("#")!=std::string::npos) {
	  // This checks for a starting macro which already starts with comma
	  _initializationStr = beginning + "\n" + inits;
	}
	else {
	  _initializationStr = beginning + ", " + inits;
	}
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
      retVal += ": " + _initializationStr;
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
