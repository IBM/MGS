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

#include "Variable.h"
#include "ConnectionCCBase.h"
#include "Generatable.h"
#include "Constants.h"
#include "AccessType.h"
#include "BaseClass.h"
#include "Class.h"
#include "ConstructorMethod.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "Attribute.h"
#include <memory>
#include <cassert>
#include <stdio.h>
#include <string.h>

Variable::Variable(const std::string& fileName) 
   : ConnectionCCBase(fileName) 
{
}

Variable::Variable(const Variable& rv)
   : ConnectionCCBase(rv) 
{
   copyContents(rv);
}

Variable Variable::operator=(const Variable& rv)
{
   if (this != &rv) {
      ConnectionCCBase::operator=(rv);
      destructContents();
      copyContents(rv);
   }
   return *this;
}

void Variable::duplicate(std::auto_ptr<Generatable>& rv) const
{
   rv.reset(new Variable(*this));
}

std::string Variable::getType() const
{
   return "Variable";
}

Variable::~Variable() 
{
   destructContents();
}

void Variable::copyContents(const Variable& rv)
{

}

void Variable::destructContents()
{

}

std::string Variable::getModuleTypeName() const
{
   return "variable";
}

void Variable::internalGenerateFiles()
{
   assert(strcmp(getName().c_str(), ""));  
   generateInstanceBase();
   generateInstance();
   generateCompCategoryBase();
   generateFactory();
   generateInAttrPSet();
   generateOutAttrPSet();
   generatePSet();
   generatePublisher();
   generateWorkUnitInstance();
   generateTriggerableCallerInstance();
   generateInstanceProxy();
}

void Variable::addCompCategoryBaseConstructorMethod(Class& instance) const
{
   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getCompCategoryBaseName()));
   constructor->addParameter("Simulation& sim");
   constructor->setInitializationStr(
      getFrameworkCompCategoryName() + "(sim)");
   constructor->setFunctionBody(getCompCategoryBaseConstructorBody());
   std::auto_ptr<Method> consToIns(constructor.release());   
   instance.addMethod(consToIns);
}

std::string Variable::getLoadedInstanceTypeName()
{
   return getCompCategoryBaseName();
}

std::string Variable::getLoadedInstanceTypeArguments()
{
   return "s";
}

void Variable::addExtraInstanceBaseMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceBaseMethods(instance);

   std::string baseName = getType() + "Base";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
  
   instance.addBaseClass(base);

   instance.addHeader("\"" + baseName + ".h\"");   

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

   // addPreConstant method
   std::auto_ptr<Method> addPreConstantMethod(new Method("addPreConstant", 
							 "void"));
   addPreConstantMethod->setVirtual();
   addPreConstantMethod->addParameter("Constant* " + PREFIX + "constant");
   addPreConstantMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreConstantMethod->setFunctionBody(getAddPreConstantFunctionBody());
   instance.addMethod(addPreConstantMethod);

   // addPreVariable method
   std::auto_ptr<Method> addPreVariableMethod(new Method("addPreVariable", 
							 "void"));
   addPreVariableMethod->setVirtual();
   addPreVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPreVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreVariableMethod->setFunctionBody(getAddPreVariableFunctionBody());
   instance.addMethod(addPreVariableMethod);

   // addPreEdge method
   std::auto_ptr<Method> addPreEdgeMethod(new Method("addPreEdge", "void"));
   addPreEdgeMethod->setVirtual();
   addPreEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPreEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreEdgeMethod->setFunctionBody(getAddPreEdgeFunctionBody());
   instance.addMethod(addPreEdgeMethod);

   // addPreNode method
   std::auto_ptr<Method> addPreNodeMethod(new Method("addPreNode", "void"));
   addPreNodeMethod->setVirtual();
   addPreNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPreNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreNodeMethod->setFunctionBody(getAddPreNodeFunctionBody());
   instance.addMethod(addPreNodeMethod);

   // add getComputeCost method
   std::auto_ptr<Method> getComputeCostMethod(
      new Method("getComputeCost", "ConnectionIncrement*"));
   getComputeCostMethod->setVirtual();
   getComputeCostMethod->setConst();
   getComputeCostMethod->setFunctionBody("#if 0\n" + TAB + "return &_computeCost;\n" + "#endif\n" + TAB + "return NULL;\n");
   instance.addMethod(getComputeCostMethod);

   addDoInitializeMethods(instance, getInstances());
}

void Variable::addExtraInstanceProxyMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceProxyMethods(instance);

   std::string baseName = getType() + "ProxyBase";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
  
   instance.addBaseClass(base);
   instance.addHeader("\"" + baseName + ".h\"");   

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

   // add getComputeCost method
   std::auto_ptr<Method> getComputeCostMethod(
      new Method("getComputeCost", "ConnectionIncrement*"));
   getComputeCostMethod->setVirtual();
   getComputeCostMethod->setConst();
   getComputeCostMethod->setFunctionBody("#if 0\n" + TAB + "return &_computeCost;\n" + "#endif\n" + TAB + "return NULL;\n");
   instance.addMethod(getComputeCostMethod);

   // add duplicate method
   instance.addDuplicateType("Variable");

   MemberContainer<DataType> empty;
}

void Variable::addExtraInstanceMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceMethods(instance);

   std::string baseName = getInstanceBaseName();
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
   instance.addBaseClass(base);

   // add duplicate method
   instance.addDuplicateType("Variable");

   instance.addHeader("\"" + baseName + ".h\"");
   instance.addStandardMethods();
}

void Variable::addExtraCompCategoryBaseMethods(Class& instance) const
{
   instance.addHeader("\"" + getInstanceName() + ".h\"");
   // for now
   instance.addHeader("<cassert>");

   // add getModelName Method
   std::auto_ptr<Method> getModelNameMethod(
      new Method("getModelName", "std::string"));
   getModelNameMethod->setVirtual();
   getModelNameMethod->setConst();
   getModelNameMethod->setFunctionBody(
      TAB + "return \"" + getName() + "\";\n");
   instance.addMethod(getModelNameMethod);

   // add getName Method
   std::auto_ptr<Method> getNameMethod(
      new Method("getName", "std::string"));
   getNameMethod->setVirtual();
   getNameMethod->setFunctionBody(
      TAB + "return \"" + getName() + "\";\n");
   instance.addMethod(getNameMethod);

   // add getDescription Method
   std::auto_ptr<Method> getDescriptionMethod(
      new Method("getDescription", "std::string"));
   getDescriptionMethod->setVirtual();
   getDescriptionMethod->setFunctionBody(
      TAB + "return \"" + getName() + "\";\n");
   instance.addMethod(getDescriptionMethod);

   // add allocateVariable Method
   std::auto_ptr<Method> allocateVariableMethod(
      new Method("allocateVariable", "Variable*") );
   allocateVariableMethod->setVirtual(true);
   std::ostringstream allocateVariableMethodFunctionBody;
   allocateVariableMethodFunctionBody 
      << TAB << "Variable* v = new " << getInstanceName() << ";\n"
      << TAB << "_variableList.push_back(v);\n"
      << TAB << "return v;\n";
   allocateVariableMethod->setFunctionBody(
      allocateVariableMethodFunctionBody.str());
   instance.addMethod(allocateVariableMethod);

   /* added by Jizhu Lu on 12/07/2005 */
   MacroConditional mpiConditional(MPICONDITIONAL);

   // added by Jizhu Lu on 04/27/2006
   CustomAttribute* demarshallerMap = new CustomAttribute("_demarshallerMap", "std::map <int, CCDemarshaller*>");
   std::auto_ptr<Attribute> demarshallerMapAptr(demarshallerMap);
   demarshallerMap->setAccessType(AccessType::PROTECTED);
   demarshallerMap->setMacroConditional(mpiConditional);
   instance.addAttribute(demarshallerMapAptr);

   CustomAttribute* demarshallerMapIter = new CustomAttribute("_demarshallerMapIter", "std::map <int, CCDemarshaller*>::iterator");
   std::auto_ptr<Attribute> demarshallerMapIterAptr(demarshallerMapIter);
   demarshallerMapIter->setAccessType(AccessType::PROTECTED);
   demarshallerMapIter->setMacroConditional(mpiConditional);
   instance.addAttribute(demarshallerMapIterAptr);
   /************************************/

   CustomAttribute* sendMap = new CustomAttribute("_sendMap", "std::map <int, ShallowArray<" + getInstanceBaseName() + "*> >");
   std::auto_ptr<Attribute> sendMapAptr(sendMap);
   sendMap->setAccessType(AccessType::PROTECTED);   
   sendMap->setMacroConditional(mpiConditional);
   instance.addAttribute(sendMapAptr);

   CustomAttribute* sendMapIter = new CustomAttribute("_sendMapIter", "std::map <int, ShallowArray<" + getInstanceBaseName() + "*> >::iterator");
   std::auto_ptr<Attribute> sendMapIterAptr(sendMapIter);
   sendMapIter->setAccessType(AccessType::PROTECTED);   
   sendMapIter->setMacroConditional(mpiConditional);
   instance.addAttribute(sendMapIterAptr);

   std::ostringstream variablesDeleteString;
   variablesDeleteString << TAB << "ShallowArray<" << getInstanceName() << "*>::iterator end1 = _variables.end();\n"
		     << TAB << "for (ShallowArray<" << getInstanceName() << "*>::iterator iter=_variables.begin(); iter!=end1; ++iter)\n"
		     << TAB << TAB << "delete (*iter);\n\n"
                     << "#ifdef HAVE_MPI\n"
                     << TAB << "std::map<int, CCDemarshaller*>::iterator end2 = _demarshallerMap.end();\n"
                     << TAB << "for (std::map<int, CCDemarshaller*>::iterator iter2=_demarshallerMap.begin(); iter2!=end2; ++iter2) {\n"
                     << TAB << TAB << "delete (*iter2).second;\n"
                     << TAB << "}\n"
                     << "#endif\n";

   /**** end of addition 12/07/2005 ****/
}

