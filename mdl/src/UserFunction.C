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

void UserFunction::duplicate(std::auto_ptr<UserFunction>& rv) const
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
   std::auto_ptr<Method> method(
      new Method(_name, "void"));
   method->setVirtual();
   method->setPureVirtual(pureVirtual);
   method->addParameter("const String& " + PREFIX + "direction");
   method->addParameter("const String& " + PREFIX + "component");
   method->addParameter("NodeDescriptor* " + PREFIX + "node");
   method->addParameter("Edge* " + PREFIX + "edge");
//   method->addParameter("Variable* " + PREFIX + "variable");
   method->addParameter("VariableDescriptor* " + PREFIX + "variable");  // modified by Jizhu Lu on 06/28/2006
   method->addParameter("Constant* " + PREFIX + "constant");
   method->addParameter(ccBase.getInAttrPSetName() + 
			"* " + PREFIX + "inAttrPset");
   method->addParameter(ccBase.getOutAttrPSetName() + 
			"* " + PREFIX + "outAttrPset");
   instance.addMethod(method);         
}

std::string UserFunction::getString() const
{
   return "\tUserFunction " + getName() + ";";
}
