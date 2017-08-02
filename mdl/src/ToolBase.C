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

#include "ToolBase.h"
#include "Generatable.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include "SyntaxErrorException.h"

#include <string>
#include <memory>
#include <iostream>
#include <sstream>

ToolBase::ToolBase(const std::string& fileName)
   : Generatable(fileName), _userInitialization(false), _name("")
{
}

void ToolBase::generate() const
{
   std::ostringstream os;
   os << getType() << " " << _name << " ";
   os << generateTitleExtra();
   os << "{\n"
      << "\t" << "Initialize (";
   MemberContainer<DataType>::const_iterator it, next
      , end = _initializeArguments.end();
   for (it = _initializeArguments.begin(); it != end; it++) {
      os << it->second->getString();
      next = it;
      next++;
      if ( next != end) {
	 os << ", ";
      }
   }
   if (_userInitialization) {
      if (_initializeArguments.size()) {
	 os << ", ";
      }
      os << "...";
   }
   os << ");\n";
   os << generateExtra();
   os << "}\n\n";
   std::cout << os.str();
}

std::string ToolBase::generateExtra() const
{
   return "";
}

std::string ToolBase::generateTitleExtra() const
{
   return "";
}

ToolBase::~ToolBase()
{
}

const std::string& ToolBase::getName() const
{
   return _name;
}

void ToolBase::setName(const std::string& name)
{
   _name = name;
}

std::string ToolBase::getModuleName() const
{
   return _name;
}

void ToolBase::generateInitializer(const std::string& type
				   , MemberContainer<DataType>& members
				   , bool userInit) 
{
   std::string fullName = PREFIX + getModuleName() + type;
   std::auto_ptr<Class> instance(new Class(fullName));

   instance->addHeader("\"SyntaxErrorException.h\"");
   instance->addHeader("\"DataItem.h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<vector>");

   instance->addAttributes(members);

   std::auto_ptr<Method> doInitMethod(
      new Method("initialize", "std::vector<DataItem*>::const_iterator") );
   doInitMethod->addParameter("const std::vector<DataItem*>& args");
   std::ostringstream doInitFunctionBody;
   if (userInit) {
      doInitFunctionBody 
	 << TAB << "if (args.size() < " << members.size() << ") {\n"
	 << TAB << TAB << "throw SyntaxErrorException(" 
	 << "\"Number of arguments should be greater than or equal to " 
	 << members.size() << "\");\n"
	 << TAB << "}\n";
   } else {
      doInitFunctionBody 
	 << TAB << "if (args.size() != " << members.size() << ") {\n"
	 << TAB << TAB << "throw SyntaxErrorException(" 
	 << "\"Number of arguments should be equal to " 
	 << members.size() << "\");\n"
	 << TAB << "}\n";
   }
   doInitFunctionBody 
      << TAB << "std::vector<DataItem*>::const_iterator " 
      + PREFIX + "currentDI = args.begin();\n";
   if (members.size() > 0) {
      MemberContainer<DataType>::const_iterator it, end = members.end();
      for (it = members.begin(); it != end; it++) {
	 doInitFunctionBody 
	    << it->second->getInitializerString(PREFIX + "currentDI");
      }
   }
   doInitFunctionBody 
      << TAB << "return " + PREFIX + "currentDI;\n"; 
   doInitMethod->setFunctionBody(doInitFunctionBody.str());
   instance->addMethod(doInitMethod);

   instance->addStandardMethods();
   _classes.push_back(instance.release());
}

void ToolBase::generateInitArgs() 
{
   generateInitializer("InitArgs", _initializeArguments, _userInitialization);
}

