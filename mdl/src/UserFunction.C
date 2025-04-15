// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UserFunction.h"
#include "Class.h"
#include "Method.h"
#include "ConnectionCCBase.h"
#include <memory>
#include <string>

UserFunction::UserFunction(const std::string& name)
   : _name(name)
{

}

void UserFunction::duplicate(std::unique_ptr<UserFunction>&& rv) const
{
   rv.reset(new UserFunction(*this));
}


UserFunction::~UserFunction()
{
}

void UserFunction::generateInstanceMethod(Class& instance, 
					  bool pureVirtual,
					  const ConnectionCCBase& ccBase) const
{
   std::unique_ptr<Method> method(
      new Method(_name, "void"));
   method->setVirtual();
   method->setPureVirtual(pureVirtual);
   method->addParameter("const CustomString& " + PREFIX + "direction");
   method->addParameter("const CustomString& " + PREFIX + "component");
   method->addParameter("NodeDescriptor* " + PREFIX + "node");
   method->addParameter("Edge* " + PREFIX + "edge");
//   method->addParameter("Variable* " + PREFIX + "variable");
   method->addParameter("VariableDescriptor* " + PREFIX + "variable");  // modified by Jizhu Lu on 06/28/2006
   method->addParameter("Constant* " + PREFIX + "constant");
   method->addParameter(ccBase.getInAttrPSetName() + 
			"* " + PREFIX + "inAttrPset");
   method->addParameter(ccBase.getOutAttrPSetName() + 
			"* " + PREFIX + "outAttrPset");
   instance.addMethod(std::move(method));         
}

std::string UserFunction::getString() const
{
   return "\tUserFunction " + getName() + ";";
}
