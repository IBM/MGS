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

#include "Attribute.h"
#include "AccessType.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>
#include <sstream>

Attribute::Attribute(int accessType)
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
      retVal = getType();
      if (isPointer()) {
	 retVal += "*";
      }
      retVal += " " + className + "::" + getName();
      if (isPointer()) {
	 retVal += " = 0";
      }
      retVal += ";\n";
   }
   return retVal;
}

std::string Attribute::getDefinition(int type) const
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

void Attribute::fillInitializer(std::string& init) const
{
   init = ""; // Important should reset to "" this is checked by the caller.
   if (getConstructorParameterName() != "") {
      init = getName() + "(" + getConstructorParameterName() + ")";
   } else if (isPointer() || isBasic()) {
      init = getName() + "(0)";	    
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
	       << tab << TAB << "rv." << getName() << "->duplicate(dup);\n"
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
