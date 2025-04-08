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

#include "PredicateFunction.h"
#include "Class.h"
#include "Method.h"
#include "ConnectionCCBase.h"
#include <memory>
#include <string>

PredicateFunction::PredicateFunction(const std::string& name)
   : _name(name)
{

}

void PredicateFunction::duplicate(std::unique_ptr<PredicateFunction>&& rv) const
{
   rv.reset(new PredicateFunction(*this));
}


PredicateFunction::~PredicateFunction()
{
}

void PredicateFunction::generateInstanceMethod(
   Class& instance, bool pureVirtual, const ConnectionCCBase& ccBase) const
{
   std::unique_ptr<Method> method(
      new Method(_name, "bool"));
   method->setVirtual();
   method->setPureVirtual(pureVirtual);
   method->addParameter("const CustomString& " + PREFIX + "direction");
   method->addParameter("const CustomString& " + PREFIX + "component");
   method->addParameter("NodeDescriptor* " + PREFIX + "node");
   method->addParameter("Edge* " + PREFIX + "edge");
   method->addParameter("VariableDescriptor* " + PREFIX + "variable");
   method->addParameter("Constant* " + PREFIX + "constant");
   method->addParameter(ccBase.getInAttrPSetName() + 
			"* " + PREFIX + "inAttrPset");
   method->addParameter(ccBase.getOutAttrPSetName() + 
			"* " + PREFIX + "outAttrPset");
   instance.addMethod(std::move(method));         
}

std::string PredicateFunction::getString() const
{
   return "\tPredicateFunction " + getName() + ";";
}
