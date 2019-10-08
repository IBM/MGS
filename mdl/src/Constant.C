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

void Constant::duplicate(std::auto_ptr<Generatable>& rv) const
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
   addSupportForMachineType(MachineType::GPU);
   assert(strcmp(getName().c_str(), ""));  
   generateType(); //e.g. CG_BiasType
   //generateInstanceBase(); // e.g. CG_Bias
   {
   auto classType = std::make_pair(Class::PrimeType::Constant, Class::SubType::BaseClass);
   bool use_classType = true;
   generateInstanceBase(use_classType, classType); //CG_Bias.h/.C
   }
   generateFactory();
   generateOutAttrPSet();
   //generatePublisher();
   {
   auto classType = std::make_pair(Class::PrimeType::Constant, Class::SubType::Publisher);
   bool use_classType = true;
   generatePublisher(use_classType, classType);//CG_BiasPublisher
   }
}

void Constant::addExtraInstanceBaseMethods(Class& instance) const
{
   /* e.g. ConstantBase */
   std::string baseName = getType() + "Base";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));  
   CustomAttribute* cusAtt = new CustomAttribute("_sim", "Simulation", AccessType::PROTECTED);
   cusAtt->setReference();
   cusAtt->setConstructorParameterNameExtra("sim");
   std::auto_ptr<Attribute> simAtt(cusAtt);
   base->addAttribute(simAtt);

   instance.addHeader("\"" + baseName + ".h\"");
   instance.addHeader("\"" + getOutAttrPSetName() + ".h\""); /* e.g. CG_BiasOutAttrPSet */

   instance.addBaseClass(base);

   // addPostVariable method
   std::auto_ptr<Method> addPostVariableMethod(new Method("addPostVariable", 
							  "void"));
   addPostVariableMethod->setVirtual();
   addPostVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPostVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostVariableMethod->setFunctionBody(getAddPostVariableFunctionBody());
   instance.addMethod(addPostVariableMethod);

   // addPostEdge method
   std::auto_ptr<Method> addPostEdgeMethod(new Method("addPostEdge", "void"));
   addPostEdgeMethod->setVirtual();
   addPostEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPostEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostEdgeMethod->setFunctionBody(getAddPostEdgeFunctionBody());
   instance.addMethod(addPostEdgeMethod);

   // addPostNode method
   std::auto_ptr<Method> addPostNodeMethod(new Method("addPostNode", "void"));
   addPostNodeMethod->setVirtual();
   addPostNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPostNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostNodeMethod->setFunctionBody(getAddPostNodeFunctionBody());
   instance.addMethod(addPostNodeMethod);

   // add CUDA GPU support
   // inline int getIndex(){return __index__;};
   std::auto_ptr<Method> getIndexMethod(new Method("getIndex", "inline int"));
   getIndexMethod->setInline();
   MacroConditional gpuConditional(GPUCONDITIONAL);
   getIndexMethod->setMacroConditional(gpuConditional);
   std::ostringstream getIndexMethodFB;   
   getIndexMethodFB 
      << TAB << TAB << TAB << "return __index__;\n";
   getIndexMethod->setFunctionBody(getIndexMethodFB.str());
   instance.addMethod(getIndexMethod);

   // add duplicate method
   instance.addDuplicateType("Constant");

   // Add out attribute PSet method
   std::auto_ptr<Method> getOutAttrParameterSetMethod(
      new Method("getOutAttrParameterSet", "void"));
   getOutAttrParameterSetMethod->setConst();
   getOutAttrParameterSetMethod->setVirtual();
   getOutAttrParameterSetMethod->addParameter(
      "std::unique_ptr<ParameterSet>& " + OUTATTRPSETNAME);
   getOutAttrParameterSetMethod->setFunctionBody(
      TAB + OUTATTRPSETNAME + ".reset(new " + getOutAttrPSetName() + "());\n");
   instance.addMethod(getOutAttrParameterSetMethod);

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
   std::auto_ptr<Attribute> simAtt(cusAtt);
   c.addAttribute(simAtt);
}

std::string Constant::getLoadedInstanceTypeArguments()
{
   return "s";
}

