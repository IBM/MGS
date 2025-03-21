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

#include "TriggeredFunction.h"
#include "Constants.h"
#include "Class.h"
#include "Method.h"
#include <sstream>
#include <cassert>

TriggeredFunction::TriggeredFunction(const std::string& name, RunType runType)
   : _name(name), _runType(runType)
{
}

std::string TriggeredFunction::getNameToCallerCodeString(
   const std::string& triggerableCallerName, 
   const std::string& className) const 
{
   std::ostringstream os;

   os << TAB << "if (\"" << _name << "\" == " << TRIGGERABLEFUNCTIONNAME << ") {\n"
      << TAB << TAB << TRIGGERABLECALLER << ".reset(new " 
      << triggerableCallerName << "(" << TRIGGERABLENDPLIST << ", &" 
      << className << "::" << getName() << ", this));\n"
      << TAB << TAB << "return " << getReturnString() << ";\n"
      << TAB << "}\n";
   return os.str();
}

TriggeredFunction::~TriggeredFunction()
{
}

std::string TriggeredFunction::getString() const
{
   std::ostringstream os;
   os << getTab() << getType() << " TriggeredFuction " << _name << ";";
   return os.str();
}

std::string TriggeredFunction::getType() const
{
   if (_runType == _SERIAL) {
      return "Serial";
   } else if (_runType == _PARALLEL) {
      return "Parallel";
   } else {
      assert(0);
   }
   return "";
}

std::string TriggeredFunction::getReturnString() const
{
   if (_runType == _SERIAL) {
      return SERIALRETURN;
   } else if (_runType == _PARALLEL) {
      return PARALLELRETURN;
   } else {
      assert(0);
   }
   return UNALTEREDRETURN;
}

void TriggeredFunction::addEventMethodToClass(Class& instance, 
					      bool pureVirtual) const
{
   std::unique_ptr<Method> eventMethod(
      new Method(getName(), "void"));
   eventMethod->addParameter("Trigger* trigger");
   eventMethod->addParameter("NDPairList* ndPairList");
   if (pureVirtual) {
      eventMethod->setPureVirtual();
   } else {
      eventMethod->setVirtual();
   }
   instance.addMethod(std::move(eventMethod));   
}
