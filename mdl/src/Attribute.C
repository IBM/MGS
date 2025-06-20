// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <memory>
#include "Attribute.h"
#include "AccessType.h"
#include "DataType.h"
#include "Constants.h"
#include "Class.h"
#include <string>
#include <vector>
#include <sstream>

Attribute::Attribute(AccessType accessType)
   : _accessType(accessType), _static(false), 
     _constructorParameterNameExtra("")
{
}

Attribute::~Attribute()
{
}

std::string Attribute::getStaticInstanceCode(
   const std::string& className) const
{
   std::string retVal = "";
   if (_static) {
      retVal = _macroConditional.getBeginning();
      retVal += getType();
      if (isPointer()) {
	 retVal += "*";
      }
      retVal += " " + className + "::" + getName();
      if (isPointer()) {
	 retVal += " = 0";
      }
      retVal += ";\n";
      retVal += _macroConditional.getEnding();
   }
   return retVal;
}

std::string Attribute::getDefinition(AccessType type) const
{
   std::ostringstream os;
   if (type == getAccessType()) {
      os << _macroConditional.getBeginning();
      os << TAB << TAB;
      if (_static) {
	 os << "static ";
      }
      os << getType();
      if (isPointer()) {
	 os << "*";
      }
      os << " " << getName() << ";\n";
      os << _macroConditional.getEnding();
   }

   return os.str();
}

void Attribute::fillInitializer(std::string& init, const Class* classObj) const
{
   init = ""; // Important should reset to "" this is checked by the caller.
   bool process_param = false;
   if (getConstructorParameterName() != "" || 
	 isPointer() || isBasic()
      )
      process_param = true;
   if (_macroConditional.getName() != "" &&
	 process_param)
      init += _macroConditional.getBeginning();
   std::string prefix=", ";

   if (_macroConditional.getName() != "" &&
	 process_param)
      init += TAB + prefix;
   if (getConstructorParameterName() != "") {
      init += getName() + "(" + getConstructorParameterName() + ")";
   } else if (isPointer() || isBasic()) {
      init += getName() + "(0)";	    
   }
   if (_macroConditional.getName() != "" &&
	 process_param)
      init += "\n" + _macroConditional.getEnding();
   if (_macroConditional.getName() != "" &&
	 process_param and classObj != 0)
   {
      if (classObj->getClassInfoPrimeType() == Class::PrimeType::Node
	    and classObj->getClassInfoSubType() == Class::SubType::BaseClassPSet
	 )
      {
	 MacroConditional tmpCond = _macroConditional;
	 tmpCond.flipCondition();
	 init += tmpCond.getBeginning();
	 init += TAB + prefix;
	 if (getConstructorParameterName() != "") {
	    init += getName() + "(" + getConstructorParameterName() + ")";
	 } else if (isPointer() || isBasic()) {
	    init += getName() + "(0)";	    
	 }
	 init += "\n" + tmpCond.getEnding();
	 tmpCond.flipCondition();
      }
   }
}

void Attribute::fillCopyInitializer(std::string& init, 
				    const std::string& copyFrom) const
{
   init = ""; // Important should reset to "" this is checked by the caller.
   if (!isDontCopy() && !(isPointer() && isOwned())) {
      init = getName() + "(" + copyFrom + getName() + ")";	    
   }
}

std::string Attribute::getCopyString(const std::string& tab)
{

   if (isDontCopy()) {
      return "";
   }

   std::ostringstream os;
   if (isPointer() && isOwned()) {
	 os << tab << "if (rv." << getName() << ") {\n";
	 if (isBasic()) {
	    os << tab << TAB << getName() << " = new " << getType() << ";\n"
	       << tab << TAB << "*" << getName() << " = *(rv." << getName() 
	       << ");\n";
	 } else {
	    os << tab << TAB << "std::unique_ptr< " << getType() << " > dup;\n"
	       << tab << TAB << "rv." << getName() << "->duplicate(std::move(dup));\n"
	       << tab << TAB << getName() << " = dup.release();\n";
	 }
	 os << tab << "} else {\n"
	    << tab << TAB << getName() << " = 0;\n"
	    << tab <<"}\n";
   } else {
      os << tab << getName() << " = rv." << getName() << ";\n";
   }
   
   return os.str();
}

std::string Attribute::getDeleteString()
{
   std::ostringstream os ;
   if(isPointer() && isOwned()) {     
	 os << TAB << "delete";
	 if (isCArray()) {
	    os << "[]";
	 }
	 os << " " << getName() << ";\n";
   }   
   return os.str();
}
