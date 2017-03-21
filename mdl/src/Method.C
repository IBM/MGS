// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Method.h"
#include "Attribute.h"
#include "AccessType.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>

Method::Method()
   : _name(""), _returnStr(""), _functionBody(""), 
     _accessType(AccessType::PUBLIC), _virtual(false),
     _pureVirtual(false), _const(false), _externC(false), 
     _externCPP(false), _inline(false), _template(false), _static(false)
{
}

Method::Method(const std::string& name, const std::string& returnStr
	       , const std::string& functionBody) 
   : _name(name), _returnStr(returnStr), _functionBody(functionBody),
     _accessType(AccessType::PUBLIC), _virtual(false), _pureVirtual(false),
     _const(false), _externC(false), _externCPP(false), _inline(false), _template(false), _static(false)
{
}

void Method::duplicate(std::auto_ptr<Method>& dup) const
{
   dup.reset(new Method(*this));
}

Method::~Method()
{
}

void Method::printSource(const std::string& className, std::ostringstream& os)
{
   if (isPureVirtual() || isInline()) {
      return;
   }
   os << _macroConditional.getBeginning();

   if (isExternC()) {
     os << "extern \"C\"\n"
	<< "{\n";
   }
   if (isExternC()) {
     os << TAB;
   }

   if (getReturnStr() != "") {
     os << _returnStr << " ";
   }
   if (!(isExternC() || isExternCPP())) {
     os << className << "::";
   }
   os << _name << "(";
   bool first = true;
   for (std::vector<std::string>::const_iterator it = _parameters.begin();
	it != _parameters.end(); it++) {
     if (first) {
       first = false;
     } else {
       os << ", ";
     }
     os << *it;
   }
   os << ") ";
   if (isConst()) {
     os << "const";
   }
   os << "\n";

   printSourceBody(os);

   if (isExternC()) {
      os << "}\n";
   } 
   os << _macroConditional.getEnding();
   os << "\n";

}

void Method::printSourceBody(std::ostringstream& os)
{
   if (isPureVirtual()) {
      return;
   }

   os << printConstructorExtra();
   if (isExternC() ) {
      os << TAB;
   }
   if (isInline() && !isExternCPP() ) {
      os << TAB << TAB;
   }
   os << "{\n";
   if (getFunctionBody() != "") {
      os << _functionBody;
   }
   if (isExternC()) {
      os << TAB;
   }
   if (isInline() && !isExternCPP() ) {
      os << TAB << TAB;
   }
   os << "}\n";
}

void Method::printDefinition(int type, std::ostringstream& os)
{
   if ((getAccessType() == type) && !isExternC() && !isExternCPP()) {
      internalPrintDefinition(TAB+TAB, os);
      if (_inline) {
	printSourceBody(os);
	os << _macroConditional.getEnding();
      }
   }
}

void Method::printExternCDefinition(std::ostringstream& os)
{
   if (isExternC()) {
      internalPrintDefinition(TAB, os);
   }
}

void Method::printExternCPPDefinition(std::ostringstream& os)
{
   if (isExternCPP()) {
      internalPrintDefinition("", os);
      if (_inline) {
	printSourceBody(os);
	os << _macroConditional.getEnding() << "\n";
      }
   }
}

std::string Method::printConstructorExtra()
{
   return "";
}

void Method::internalPrintDefinition(const std::string& tab
				     , std::ostringstream& os)
{
   os << _macroConditional.getBeginning();
   os << tab;
   if (isVirtual() || isPureVirtual()) {
      os << "virtual" << " ";
   }      
   if (isStatic()) {
     os << "static ";
   }
   if (isTemplate()) {
     std::string s;
     getTemplateParametersString(s);
     os << "template" << s << " ";
   }
   if (_returnStr != "") {
      os << _returnStr << " ";
   }
   os << _name << "(";
   bool first = true;
   for (std::vector<std::string>::const_iterator it = _parameters.begin();
	it != _parameters.end(); it++) {
      if (first) {
	 first = false;
      } else {
	 os << ", ";
      }
      os << *it;
   }
   os << ")";
   if (isConst()) {
      os << " const";
   }
   if (isPureVirtual()) {
      os << "=0";
      if (isInline()) os << ";";
   }
   if (!isInline()) os << ";";
   os << "\n";
   if (!isInline()) os << _macroConditional.getEnding();
}

void Method::getTemplateParametersString(std::string& s) 
{
  s = " <";
  std::vector<std::string>::iterator iter = _templateParameters.begin();
  std::vector<std::string>::iterator end = _templateParameters.end();
  while (iter != end) {
    s += (*iter);
    if (++iter != end) s += ", ";
  }
  s += ">";
}

