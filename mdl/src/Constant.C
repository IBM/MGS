// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Constant.h"
#include "InterfaceImplementorBase.h"
#include "Generatable.h"
#include "Class.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "Attribute.h"
#include "BaseClass.h"
#include <memory>
#include <cassert>
#include <cstring>

Constant::Constant(const std::string& fileName) 
   : InterfaceImplementorBase(fileName) 
{

}

void Constant::duplicate(std::unique_ptr<Generatable>&& rv) const
{
   rv.reset(new Constant(*this));
}

std::string Constant::getType() const
{
   return "Constant";
}

Constant::~Constant() 
{
}

std::string Constant::getModuleTypeName() const
{
   return "constant";
}

void Constant::internalGenerateFiles()
{
   assert(strcmp(getName().c_str(), ""));  
   generateType();
   generateInstanceBase();
   generateFactory();
   generateOutAttrPSet();
   generatePublisher();
}

void Constant::addExtraInstanceBaseMethods(Class& instance) const
{
   std::string baseName = getType() + "Base";
   std::unique_ptr<BaseClass> base(new BaseClass(baseName));  
   CustomAttribute* cusAtt = new CustomAttribute("_sim", "Simulation", AccessType::PROTECTED);
   cusAtt->setReference();
   cusAtt->setConstructorParameterNameExtra("sim");
   std::unique_ptr<Attribute> simAtt(cusAtt);
   base->addAttribute(std::move(simAtt));

   instance.addHeader("\"" + baseName + ".h\"");
   instance.addHeader("\"" + getOutAttrPSetName() + ".h\"");

   instance.addBaseClass(std::move(base));

   // addPostVariable method
   std::unique_ptr<Method> addPostVariableMethod(new Method("addPostVariable", 
							  "void"));
   addPostVariableMethod->setVirtual();
   addPostVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPostVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostVariableMethod->setFunctionBody(getAddPostVariableFunctionBody());
   instance.addMethod(std::move(addPostVariableMethod));

   // addPostEdge method
   std::unique_ptr<Method> addPostEdgeMethod(new Method("addPostEdge", "void"));
   addPostEdgeMethod->setVirtual();
   addPostEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPostEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostEdgeMethod->setFunctionBody(getAddPostEdgeFunctionBody());
   instance.addMethod(std::move(addPostEdgeMethod));

   // addPostNode method
   std::unique_ptr<Method> addPostNodeMethod(new Method("addPostNode", "void"));
   addPostNodeMethod->setVirtual();
   addPostNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPostNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostNodeMethod->setFunctionBody(getAddPostNodeFunctionBody());
   instance.addMethod(std::move(addPostNodeMethod));

   // add duplicate method
   instance.addDuplicateType("Constant");

   // Add out attribute PSet method
   std::unique_ptr<Method> getOutAttrParameterSetMethod(
      new Method("getOutAttrParameterSet", "void"));
   getOutAttrParameterSetMethod->setConst();
   getOutAttrParameterSetMethod->setVirtual();
   getOutAttrParameterSetMethod->addParameter(
      "std::unique_ptr<ParameterSet>& " + OUTATTRPSETNAME);
   getOutAttrParameterSetMethod->setFunctionBody(
      TAB + OUTATTRPSETNAME + ".reset(new " + getOutAttrPSetName() + "());\n");
   instance.addMethod(std::move(getOutAttrParameterSetMethod));

   addDoInitializeMethods(instance, getInstances());
}

std::string Constant::getInstanceNameForTypeArguments() const
{
   return "_sim";
}


void Constant::addGenerateTypeClassAttribute(Class& c) const
{
   CustomAttribute* cusAtt = new CustomAttribute("_sim", "Simulation", AccessType::PROTECTED);
   cusAtt->setReference();
   cusAtt->setConstructorParameterNameExtra("sim");
   std::unique_ptr<Attribute> simAtt(cusAtt);
   c.addAttribute(simAtt);
}

std::string Constant::getLoadedInstanceTypeArguments()
{
   return "s";
}

