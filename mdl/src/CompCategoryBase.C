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

#include "CompCategoryBase.h"
#include "InterfaceImplementorBase.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "MemberToInterface.h"
#include "InternalException.h"
#include "GeneralException.h"
#include "Constants.h"
#include "StructType.h"
#include "Class.h"
#include "Method.h"
#include "ConstructorMethod.h"
#include "Attribute.h"
#include "CustomAttribute.h"
#include "BaseClass.h"
#include "Interface.h"
#include "DataType.h"
#include "Phase.h"
#include "Utility.h"
#include "MacroConditional.h"
#include "TypeDefinition.h"
#include "FriendDeclaration.h"
#include "ConnectionIncrement.h"
#include <memory>
#include <sstream>
#include <iostream>
#include <cassert>
#include <list>
#include <stdio.h>
#include <string.h>

CompCategoryBase::CompCategoryBase(const std::string& fileName) 
   : InterfaceImplementorBase(fileName), _inAttrPSet(0), 
     _triggeredFunctions(0), _connectionIncrement(0)
{
}

CompCategoryBase::CompCategoryBase(const CompCategoryBase& rv)
   : InterfaceImplementorBase(rv)
{
   copyOwnedHeap(rv);
}

CompCategoryBase& CompCategoryBase::operator=(const CompCategoryBase& rv)
{
   if (this != &rv) {
      InterfaceImplementorBase::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void CompCategoryBase::releaseInAttrPSet(std::auto_ptr<StructType>& iap)
{
   iap.reset(_inAttrPSet);
   _inAttrPSet = 0;
}

void CompCategoryBase::setInAttrPSet(std::auto_ptr<StructType>& iap)
{
   delete _inAttrPSet;
   _inAttrPSet = iap.release();
}

std::string CompCategoryBase::generateExtra() const
{
   std::ostringstream os;
   if (_instancePhases || getInterfaces().size() > 0) {
      os << "\n";
   }
   if (_instancePhases) {
      std::vector<Phase*>::iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 os << "\t" << (*it)->getGenerateString() << ";\n";
      }
      os << "\n";
   }
   if (_triggeredFunctions) {
      std::vector<TriggeredFunction*>::const_iterator it, 
	 end = _triggeredFunctions->end();
      for (it = _triggeredFunctions->begin(); it != end; ++it) {
	 os << (*it)->getString() << "\n";
      }
   }

   os << _inAttrPSet->getInAttrPSetStr();  
   return os.str();
}

void CompCategoryBase::isDuplicatePhase(const Phase* phase) 
{
   if (_instancePhases) {
      if (mdl::findInPhases(phase->getName(), *_instancePhases)) {
	 std::ostringstream os;
	 os << "phase " << phase->getName() << " is already included as an " 
	    << phase->getType();
	 throw DuplicateException(os.str());
      }
   }
}

void CompCategoryBase::isDuplicateTriggeredFunction(const std::string& name)
{
   if (_triggeredFunctions) {
      if (mdl::findInTriggeredFunctions(name, *_triggeredFunctions)) {
	 std::ostringstream os;
	 os << "Triggered function " << name << " is already included.";
	 throw DuplicateException(os.str());
      }
   }
}

CompCategoryBase::~CompCategoryBase() {
   destructOwnedHeap();
}

void CompCategoryBase::copyOwnedHeap(const CompCategoryBase& rv)
{
   if (rv._instancePhases) {
      _instancePhases = new std::vector<Phase*>();
      std::vector<Phase*>::const_iterator it, 
	 end = rv._instancePhases->end();
      std::auto_ptr<Phase> dup;   
      for (it = rv._instancePhases->begin(); it != end; ++it) {
	 (*it)->duplicate(dup);
	 _instancePhases->push_back(dup.release());
      }
   } else {
      _instancePhases = 0;
   }
   if (rv._inAttrPSet) {
      std::auto_ptr<StructType> dup;
      rv._inAttrPSet->duplicate(dup);
      _inAttrPSet = dup.release();
   } else {
      _inAttrPSet = 0;
   }
   if (rv._triggeredFunctions) {
      _triggeredFunctions = new std::vector<TriggeredFunction*>();
      std::vector<TriggeredFunction*>::const_iterator it
	 , end = rv._triggeredFunctions->end();
      std::auto_ptr<TriggeredFunction> dup;   
      for (it = rv._triggeredFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(dup);
	 _triggeredFunctions->push_back(dup.release());
      }
   } else {
      _triggeredFunctions = 0;
   }
   if (rv._connectionIncrement) {
      std::auto_ptr<ConnectionIncrement> dup;
      rv._connectionIncrement->duplicate(dup);
      _connectionIncrement = dup.release();
   } else {
      _connectionIncrement = 0;
   }
}

void CompCategoryBase::destructOwnedHeap()
{
   if (_instancePhases) {
      std::vector<Phase*>::iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _instancePhases;
      _instancePhases = 0;
   }
   delete _inAttrPSet;
   if (_triggeredFunctions) {
      std::vector<TriggeredFunction*>::iterator it, 
	 end = _triggeredFunctions->end();
      for (it = _triggeredFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _triggeredFunctions;
      _triggeredFunctions = 0;
   }

}

void CompCategoryBase::setTriggeredFunctions(
   std::auto_ptr<std::vector<TriggeredFunction*> >& triggeredFunction) 
{
   delete _triggeredFunctions;
   _triggeredFunctions = triggeredFunction.release();
}

void CompCategoryBase::generateInAttrPSet()
{
   std::auto_ptr<Class> instance;
   createPSetClass(instance, _inAttrPSet->_members, "InAttr");
   _classes.push_back(instance.release());
}

void CompCategoryBase::generatePSet()
{
   std::auto_ptr<Class> instance;
   createPSetClass(instance, getInstances());
   _classes.push_back(instance.release());
}

void CompCategoryBase::addExtraInstanceBaseMethods(Class& instance) const
{
   instance.addClass(getCompCategoryBaseName());
   instance.addHeader("\"" + getPSetName() + ".h\"");
   instance.addHeader("\"" + getInAttrPSetName() + ".h\"");
   instance.addHeader("\"" + getOutAttrPSetName() + ".h\"");


   // Add the friends
   FriendDeclaration ccBaseFriend(getCompCategoryBaseName(), MPICONDITIONAL);
   instance.addFriendDeclaration(ccBaseFriend);

   // initialize method
   std::auto_ptr<Method> initializeMethod(new Method("initialize", "void"));
   initializeMethod->setVirtual();
   initializeMethod->addParameter("ParameterSet* " + PREFIX + "initPSet");
   std::ostringstream initializeFB;
   if (getInstances().size() > 0) {
      MemberContainer<DataType>::const_iterator it, end = getInstances().end();
      initializeFB 
	 << TAB << getPSetName() << "* " << PREFIX << "pset = dynamic_cast<" 
	 << getPSetName() << "*>"
	 << "(" << PREFIX << "initPSet);\n"; 
      for (it = getInstances().begin(); it != end; ++it){
	 initializeFB
	    << TAB << it->first << " = " << PREFIX << "pset->" 
	    << it->first << ";\n";
      }
   }
   initializeMethod->setFunctionBody(initializeFB.str());
   instance.addMethod(initializeMethod);

   // Add the phase methods
   if(_instancePhases) {
      std::vector<Phase*>::iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 (*it)->generateVirtualUserMethod(instance);
      }
   }

   // Add the triggerable event methods
   if(_triggeredFunctions) {
      addTriggeredFunctionMethods(instance, *_triggeredFunctions, true);
   }

   addCreateTriggerableCallerMethod(instance, _triggeredFunctions,
				    getTriggerableCallerInstanceName());

   instance.addExtraSourceHeader(
      "\"" + getTriggerableCallerInstanceName() + ".h\"");

   // Add initialization PSet method
   std::auto_ptr<Method> getInitializationParameterSetMethod(
      new Method("getInitializationParameterSet", "void"));
   getInitializationParameterSetMethod->setConst();
   getInitializationParameterSetMethod->setVirtual();
   getInitializationParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& initPSet");
   getInitializationParameterSetMethod->setFunctionBody(
      TAB + "initPSet.reset(new " + getPSetName() + "());\n");
   instance.addMethod(getInitializationParameterSetMethod);

   // Add in attribute PSet method
   std::auto_ptr<Method> getInAttrParameterSetMethod(
      new Method("getInAttrParameterSet", "void"));
   getInAttrParameterSetMethod->setConst();
   getInAttrParameterSetMethod->setVirtual();
   getInAttrParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& " + INATTRPSETNAME);
   getInAttrParameterSetMethod->setFunctionBody(
      TAB + INATTRPSETNAME + ".reset(new " + getInAttrPSetName() + "());\n");
   instance.addMethod(getInAttrParameterSetMethod);

   // Add out attribute PSet method
   std::auto_ptr<Method> getOutAttrParameterSetMethod(
      new Method("getOutAttrParameterSet", "void"));
   getOutAttrParameterSetMethod->setConst();
   getOutAttrParameterSetMethod->setVirtual();
   getOutAttrParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& " + OUTATTRPSETNAME);
   getOutAttrParameterSetMethod->setFunctionBody(
      TAB + OUTATTRPSETNAME + ".reset(new " + getOutAttrPSetName() + "());\n");
   instance.addMethod(getOutAttrParameterSetMethod);
}

void CompCategoryBase::addExtraInstanceProxyMethods(Class& instance) const
{
   // Add the friends
   FriendDeclaration ccBaseFriend(getCompCategoryBaseName());
   instance.addFriendDeclaration(ccBaseFriend);

   if (!strcmp(getType().c_str(), "Variable")) {
     instance.addHeader("\"" + getPSetName() + ".h\"");
     instance.addHeader("\"" + getInAttrPSetName() + ".h\"");
     instance.addHeader("\"" + getOutAttrPSetName() + ".h\"");

   // Add initialization PSet method
     std::auto_ptr<Method> getInitializationParameterSetMethod(
       new Method("getInitializationParameterSet", "void"));
     getInitializationParameterSetMethod->setConst();
     getInitializationParameterSetMethod->setVirtual();
     getInitializationParameterSetMethod->addParameter(
						       "std::auto_ptr<ParameterSet>& initPSet");
     getInitializationParameterSetMethod->setFunctionBody(
							  TAB + "initPSet.reset(new " + getPSetName() + "());\n");
     instance.addMethod(getInitializationParameterSetMethod);

   // Add in attribute PSet method
     std::auto_ptr<Method> getInAttrParameterSetMethod(
       new Method("getInAttrParameterSet", "void"));
     getInAttrParameterSetMethod->setConst();
     getInAttrParameterSetMethod->setVirtual();
     getInAttrParameterSetMethod->addParameter(
					       "std::auto_ptr<ParameterSet>& " + INATTRPSETNAME);
     getInAttrParameterSetMethod->setFunctionBody(
       TAB + INATTRPSETNAME + ".reset(new " + getInAttrPSetName() + "());\n");
     instance.addMethod(getInAttrParameterSetMethod);

   // Add out attribute PSet method
     std::auto_ptr<Method> getOutAttrParameterSetMethod(
       new Method("getOutAttrParameterSet", "void"));
     getOutAttrParameterSetMethod->setConst();
     getOutAttrParameterSetMethod->setVirtual();
     getOutAttrParameterSetMethod->addParameter(
						"std::auto_ptr<ParameterSet>& " + OUTATTRPSETNAME);
     getOutAttrParameterSetMethod->setFunctionBody(
       TAB + OUTATTRPSETNAME + ".reset(new " + getOutAttrPSetName() + "());\n");
     instance.addMethod(getOutAttrParameterSetMethod);
   }
   instance.addClass(getCompCategoryBaseName());

}

void CompCategoryBase::generateInstance()
{
   std::auto_ptr<Class> instance(new Class(getInstanceName()));

   instance->setUserCode();

   // Add the phase methods
   if(_instancePhases) {
      std::vector<Phase*>::iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 (*it)->generateUserMethod(*instance.get());
      }
   }

   // Add the triggerable event methods
   if(_triggeredFunctions) {
      addTriggeredFunctionMethods(*(instance.get()), 
				  *_triggeredFunctions, false);
   }

   addExtraInstanceMethods(*(instance.get()));
   _classes.push_back(instance.release());  
}

void CompCategoryBase::generateCompCategoryBase()
{
   std::auto_ptr<Class> instance(new Class(getCompCategoryBaseName()));

   std::auto_ptr<BaseClass> base(new BaseClass(getFrameworkCompCategoryName()));   
   instance->addBaseClass(base);
 
   //instance->addHeader("\"Edge.h\"");
   instance->addHeader("\"Simulation.h\"");
   instance->addHeader("\"" + getFrameworkCompCategoryName() + ".h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<deque>");
   instance->addHeader("<string>");
   instance->addHeader("<iostream>");
   instance->addHeader("<set>");
   instance->addHeader("\"" + getPSetName() + ".h\"");
   instance->addHeader("\"" + getInAttrPSetName() + ".h\"");
   instance->addHeader("\"" + getOutAttrPSetName() + ".h\"");
   instance->addHeader("\"" + getWorkUnitInstanceName() + ".h\"");
   instance->addHeader("\"ConnectionIncrement.h\"", MPICONDITIONAL);
   instance->addHeader("\"MemPattern.h\"", MPICONDITIONAL);
   instance->addClass("NDPairList");
   instance->addClass(getInstanceBaseName());

   addCompCategoryBaseConstructorMethod(*(instance.get()));

   // Add initialization PSet method
   std::auto_ptr<Method> getInitializationParameterSetMethod(
      new Method("getInitializationParameterSet", "void"));
   getInitializationParameterSetMethod->setVirtual();
   getInitializationParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& initPSet");
   getInitializationParameterSetMethod->setFunctionBody(
      TAB + "initPSet.reset(new " + getPSetName() + "());\n");
   instance->addMethod(getInitializationParameterSetMethod);

   // Add in attribute PSet method
   std::auto_ptr<Method> getInAttrParameterSetMethod(
      new Method("getInAttrParameterSet", "void"));
   getInAttrParameterSetMethod->setVirtual();
   getInAttrParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& " + INATTRPSETNAME);
   getInAttrParameterSetMethod->setFunctionBody(
      TAB + INATTRPSETNAME + ".reset(new " + getInAttrPSetName() + "());\n");
   instance->addMethod(getInAttrParameterSetMethod);

   // Add out attribute PSet method
   std::auto_ptr<Method> getOutAttrParameterSetMethod(
      new Method("getOutAttrParameterSet", "void"));
   getOutAttrParameterSetMethod->setVirtual();
   getOutAttrParameterSetMethod->addParameter(
      "std::auto_ptr<ParameterSet>& " + OUTATTRPSETNAME);
   getOutAttrParameterSetMethod->setFunctionBody(
      TAB + OUTATTRPSETNAME + ".reset(new " + getOutAttrPSetName() + "());\n");
   instance->addMethod(getOutAttrParameterSetMethod);

   // Add the instance phase caller methods
   if(_instancePhases) {
      std::vector<Phase*>::iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 (*it)->generateInstancePhaseMethod(
	    *instance.get(), getInstanceName(), getType());
      }
   }  

   // Add getWorkUnits method 
   std::string phaseName = "phaseName";
   std::string workUnits = "workUnits";
   std::auto_ptr<Method> getWorkUnitsMethod(
      new Method("getWorkUnits", "void"));   
   getWorkUnitsMethod->setVirtual();
   getWorkUnitsMethod->setFunctionBody(
      createGetWorkUnitsMethodBody(phaseName, workUnits));
   instance->addMethod(getWorkUnitsMethod);

   if (!strcmp(getType().c_str(), "Node") || !strcmp(getType().c_str(), "Variable")) {
      MacroConditional mpiConditional(MPICONDITIONAL);
   
      // Add addToSendMap method 
      std::string secondParam;
      std::string firstParam = "toPartitionId";
      if (!strcmp(getType().c_str(), "Node"))
         secondParam = "node";
      else if (!strcmp(getType().c_str(), "Variable"))
         secondParam = "variable";
      std::auto_ptr<Method> addToSendMap(
         new Method("addToSendMap", "void"));   
      addToSendMap->addParameter("int " + firstParam);
      addToSendMap->addParameter(getType() + "* " + secondParam);
      addToSendMap->setVirtual();
      addToSendMap->setFunctionBody(createAddNodeMethodBody(firstParam, secondParam));
      addToSendMap->setMacroConditional(mpiConditional);
      instance->addMethod(addToSendMap);
   
      // Add allocateProxy method 
      firstParam = "fromPartitionId";
      if (!strcmp(getType().c_str(), "Node"))
         secondParam = "nd";
      else if (!strcmp(getType().c_str(), "Variable"))
         secondParam = "vd";
      std::auto_ptr<Method> allocateProxy(
         new Method("allocateProxy", "void"));   
      allocateProxy->addParameter("int " + firstParam);
      allocateProxy->addParameter(getType() + "Descriptor* " + secondParam);
      allocateProxy->setVirtual();
      allocateProxy->setFunctionBody(createAllocateProxyMethodBody(firstParam, secondParam));
      allocateProxy->setMacroConditional(mpiConditional);
      instance->addMethod(allocateProxy);
   }

   if (!strcmp(getType().c_str(), "Node") || !strcmp(getType().c_str(), "Variable"))
      addDistributionCodeToCC(*(instance.get()));

   addExtraCompCategoryBaseMethods(*(instance.get()));
   _classes.push_back(instance.release());  
}

void CompCategoryBase::generateCompCategory()
{
   // placeholder
   std::auto_ptr<Class> instance(new Class(getCompCategoryName()));

   instance->setUserCode();

   std::string baseName = getCompCategoryBaseName();
   std::auto_ptr<BaseClass> base(
      new BaseClass(baseName));
   instance->addBaseClass(base);

   instance->addHeader("\"" + baseName + ".h\"");
   instance->addClass("NDPairList");

   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getCompCategoryName()));
   constructor->addParameter("Simulation& sim");
   constructor->addParameter("const std::string& modelName");
   constructor->addParameter("const NDPairList& ndpList");
   constructor->setInitializationStr(
      getCompCategoryBaseName() + "(sim, modelName, ndpList)");
   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);

   addExtraCompCategoryMethods(*(instance.get()));
   _classes.push_back(instance.release());  
}

void CompCategoryBase::generateTriggerableCallerInstance()
{
   generateTriggerableCallerCommon(getInstanceBaseName());
}

void CompCategoryBase::generateWorkUnitInstance()
{
   generateWorkUnitCommon("Instance", getType() + "PartitionItem",
			  getCompCategoryBaseName());
}

void CompCategoryBase::generateWorkUnitCommon(const std::string& workUnitType, 
					      const std::string& argumentType, 
					      const std::string& compCatName)
{
   std::string fullName = getWorkUnitCommonName(workUnitType);
   std::auto_ptr<Class> instance(new Class(fullName));

   std::string baseName = "WorkUnit";
   std::auto_ptr<BaseClass> base(
      new BaseClass(baseName));
   instance->addBaseClass(base);

   std::string computeStateAttName = 
      "(" + compCatName + "::*_computeState) (" + argumentType; 
   if (argumentType != "") {
      computeStateAttName += "*, "; 
   }
   computeStateAttName += "RNG&";
   computeStateAttName += ")"; 

   std::string computeStatePar = 
      "(" + compCatName + "::*computeState) (" + argumentType; 
   if (argumentType != "") {
      computeStatePar += "*, "; 
   }
   computeStatePar += "RNG&";
   computeStatePar += ")"; 

   instance->addHeader("\"" + baseName + ".h\"");
   instance->addClass(compCatName);
   instance->addHeader("\"rndm.h\"");
   
   std::auto_ptr<Attribute> attCup;

   if (argumentType != "") {
      instance->addHeader("\"" + argumentType + ".h\"");

      CustomAttribute* argAtt = new CustomAttribute("_arg", argumentType, AccessType::PRIVATE);
      argAtt->setPointer();
      attCup.reset(argAtt);
      instance->addAttribute(attCup);
   }

   CustomAttribute* compCategoryAtt = new CustomAttribute(
      "_compCategory", compCatName, AccessType::PRIVATE);
   compCategoryAtt->setPointer();
   attCup.reset(compCategoryAtt);
   instance->addAttribute(attCup);

   CustomAttribute* computeStateAtt = new CustomAttribute(
      computeStateAttName, "void", 
      AccessType::PRIVATE);
   attCup.reset(computeStateAtt);
   instance->addAttribute(attCup);

   CustomAttribute* rngAtt = new CustomAttribute(
      "_rng", "RNG",
      AccessType::PRIVATE);
   attCup.reset(rngAtt);
   instance->addAttribute(attCup);

   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(fullName));
   if (argumentType != "") {
      constructor->addParameter(argumentType + "* arg");
   } 
   constructor->addParameter("void " + computeStatePar);
   constructor->addParameter(compCatName + "* compCategory");
   std::string initializationStr = "WorkUnit(), ";
   if (argumentType != "") {
      initializationStr += "_arg(arg), ";
   }
   initializationStr 
      += "_compCategory(compCategory), _computeState(computeState)";
   constructor->setInitializationStr(initializationStr);
   std::ostringstream constructorFB;
   constructorFB << TAB << "_rng.reSeed(urandom(_compCategory->getSimulation().getWorkUnitRandomSeedGenerator()), _compCategory->getSimulation().getRank());\n";      
   constructor->setFunctionBody(constructorFB.str());
   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);

   // execute 
   std::auto_ptr<Method> executeMethod(
      new Method("execute", "void"));
   executeMethod->setVirtual();
   std::ostringstream executeFB;   
   executeFB
      << TAB << "(*_compCategory.*_computeState)(";
   if (argumentType != "") {
      executeFB
	 << "_arg, ";
   }
   executeFB
     << "_rng";
   executeFB
      << ");\n";
   executeMethod->setFunctionBody(executeFB.str());
   instance->addMethod(executeMethod);

   // Don't add the standard methods
   instance->addBasicDestructor();
   _classes.push_back(instance.release());  

}

void CompCategoryBase::generateTriggerableCallerCommon(
   const std::string& modelType)
{
   std::string fullName = getTriggerableCallerCommonName(modelType);
   std::auto_ptr<Class> instance(new Class(fullName));

   std::string baseName = "TriggerableCaller";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
   instance->addBaseClass(base);

   std::string computeStateAttName = 
      "(" + modelType + "::*_function) (Trigger*, NDPairList*)"; 

   std::string computeStatePar = 
      "(" + modelType + "::*function) (Trigger*, NDPairList*)"; 

   instance->addHeader("\"" + baseName + ".h\"");
   instance->addClass(modelType);
   
   std::auto_ptr<Attribute> attCup;

   CustomAttribute* computeStateAtt = new CustomAttribute(
      computeStateAttName, "void", 
      AccessType::PRIVATE);
   attCup.reset(computeStateAtt);
   instance->addAttribute(attCup);

   CustomAttribute* compCategoryAtt = new CustomAttribute(
      "_triggerable", modelType, AccessType::PRIVATE);
   compCategoryAtt->setPointer();
   attCup.reset(compCategoryAtt);
   instance->addAttribute(attCup);

   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(fullName));
   constructor->addParameter("NDPairList* ndPairList");
   constructor->addParameter("void " + computeStatePar);
   constructor->addParameter(modelType + "* triggerable");
   std::string initializationStr = "TriggerableCaller(ndPairList), ";
   initializationStr 
      += "_function(function), _triggerable(triggerable)";
   constructor->setInitializationStr(initializationStr);
   std::ostringstream constructorFB;
   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);

   // execute 
   std::auto_ptr<Method> eventMethod(
      new Method("event", "void"));
   eventMethod->setVirtual();
   eventMethod->addParameter("Trigger* trigger");
   eventMethod->setFunctionBody(
      TAB + "(*_triggerable.*_function)(trigger, _ndPairList);\n");
   instance->addMethod(eventMethod);

   // getTriggerable 
   std::auto_ptr<Method> getTriggerableMethod(
      new Method("getTriggerable", "Triggerable*"));
   getTriggerableMethod->setVirtual();
   getTriggerableMethod->setFunctionBody(
      TAB + "return _triggerable;\n");
   instance->addMethod(getTriggerableMethod);

//    // Don't add the standard methods
//    instance->addBasicDestructor();
   instance->addStandardMethods();
   _classes.push_back(instance.release());  
}

std::string CompCategoryBase::createAddNodeMethodBody(std::string firstParam, std::string secondParam) const
{
   std::ostringstream os;
   if (!strcmp(getType().c_str(), "Node"))
      os << TAB << getInstanceBaseName() << "* local" + getType() +" = dynamic_cast<" << getInstanceBaseName() << "*>(node);\n";
   else if (!strcmp(getType().c_str(), "Variable"))
      os << TAB << getInstanceBaseName() << "* local" + getType() +" = dynamic_cast<" << getInstanceBaseName() << "*>(variable);\n";
   os << TAB << "assert(local" + getType() + ");\n";

   os << TAB << "ShallowArray<" << getInstanceBaseName() << "*>::iterator it = _sendMap[" + firstParam + "].begin();\n";
   os << TAB << "ShallowArray<" << getInstanceBaseName() << "*>::iterator end = _sendMap[" + firstParam + "].end();\n";

   os << TAB << "bool found = false;\n";
   os << TAB << "for (; it != end; ++it) {\n";
   os << TAB << TAB << "if ((*it) == local" + getType() + ") {\n";
   os << TAB << TAB << TAB << "found = true;\n";
   os << TAB << TAB << TAB << "break;\n";
   os << TAB << TAB << "}\n";
   os << TAB << "}\n";
   os << TAB << "if (found == false)\n";
   os << TAB << TAB << "_sendMap[" + firstParam + "].push_back(local" + getType() + ");\n";
   return os.str();
}

std::string CompCategoryBase::createAllocateProxyMethodBody(std::string firstParam, std::string secondParam) const
{
   std::ostringstream os;
   os << TAB << "CCDemarshaller* ccd = findDemarshaller(fromPartitionId);\n";
   os << TAB << getType() << "ProxyBase* proxy = ccd->addDestination();\n";
   os << TAB << "proxy->set" + getType() + "Descriptor(" + secondParam + ");\n";
   if (!strcmp(getType().c_str(), "Variable")) os << TAB << "proxy->setVariableType(this);\n";
   os << TAB << secondParam + "->set" + getType() + "(proxy);\n";               
   return os.str();
}

std::string CompCategoryBase::createGetWorkUnitsMethodBody(
   const std::string& phaseName, const std::string& workUnits) const
{
   std::ostringstream os;
   if (_instancePhases) {
      os << createGetWorkUnitsMethodCommonBody(
	 phaseName, workUnits, getInstanceBaseName(), *_instancePhases);
   }
   return os.str();
}

std::string CompCategoryBase::createGetWorkUnitsMethodCommonBody(
   const std::string& phaseName, const std::string& workUnits, 
   const std::string& instanceName, const std::vector<Phase*>& phases) const
{
   std::ostringstream os;
   std::vector<Phase*>::const_iterator it, end = phases.end();
   for (it = phases.begin(); it != end; ++it) {
      os << TAB << "{\n"
	 << (*it)->getWorkUnitsMethodBody(TAB + TAB, workUnits, instanceName,
					  getType());
      os << TAB << "}\n";
   }
   return os.str();
}     

// Will be implemented in derived classes.
std::string CompCategoryBase::getCompCategoryBaseConstructorBody() const
{
   std::ostringstream os;
   if (_instancePhases) {
      std::vector<Phase*>::const_iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 os << (*it)->getInitializePhaseMethodBody();
      }
   }
   return os.str();
}

void CompCategoryBase::addTriggeredFunctionMethods(
   Class& instance, 
   const std::vector<TriggeredFunction*>& functions, bool pureVirtual) const
{
   std::vector<TriggeredFunction*>::const_iterator it, end = functions.end();
   for(it = functions.begin(); it != end; ++it) {
      (*it)->addEventMethodToClass(instance, pureVirtual);
   }
}

void CompCategoryBase::addCreateTriggerableCallerMethod(
   Class& instance, 
   const std::vector<TriggeredFunction*>* functions,
   const std::string& triggerableCallerName) const
{
   std::auto_ptr<Method> method(
      new Method("createTriggerableCaller", EVENTTYPE));
   method->setAccessType(AccessType::PROTECTED);
   method->setVirtual();
   method->addParameter("const std::string& " + TRIGGERABLEFUNCTIONNAME);
   method->addParameter("NDPairList* " + TRIGGERABLENDPLIST);
   method->addParameter("std::auto_ptr<TriggerableCaller>& " + 
			TRIGGERABLECALLER);
   std::ostringstream os;

   if (functions) {
      std::vector<TriggeredFunction*>::const_iterator it, 
	 end = functions->end();
      for(it = functions->begin(); it != end; ++it) {
	 os << (*it)->getNameToCallerCodeString(triggerableCallerName, 
						instance.getName());
      }
   }
   os << TAB 
      << "throw SyntaxErrorException(" << TRIGGERABLEFUNCTIONNAME 
      << " + \" is not defined in " << getInstanceName() 
      << " as a Triggerable function.\");\n"
      << TAB << "return " << UNALTEREDRETURN << ";\n";
   method->setFunctionBody(os.str());
   instance.addMethod(method);   
}

std::string CompCategoryBase::getAddVariableNamesForPhaseFB() const
{
   if (_instancePhases) {
      std::ostringstream os;
      std::vector<Phase*>::const_iterator it, end = _instancePhases->end();
      for (it = _instancePhases->begin(); it != end; ++it) {
	 os << (*it)->getAddVariableNamesForPhase(TAB);
      }
      return os.str();
   } 
   return "";      
}

std::string CompCategoryBase::getSetDistributionTemplatesFB() const
{
   std::ostringstream os;
   os << TAB << SENDTEMPLATES <<"[\"FLUSH_LENS\"] = &" + getInstanceBaseName()
      << "::CG_send_FLUSH_LENS;\n"
      << TAB << GETSENDTYPETEMPLATES <<"[\"FLUSH_LENS\"] = &" + getInstanceBaseName()
      << "::CG_getSendType_FLUSH_LENS;\n"
      << TAB << "std::map<std::string, Phase*>::iterator it, end = _phaseMappings.end();\n"
      << TAB << "for (it = _phaseMappings.begin(); it != end; ++it) {\n"
      << getTemplateFillerCode()
      << TAB << "}\n";
   return os.str();
}

std::string CompCategoryBase::getResetSendProcessIdIteratorsFB() const
{
   std::ostringstream os;
   os << TAB  << "_sendMapIter=_sendMap.begin();\n";
   return os.str();
}

std::string CompCategoryBase::getGetSendNextProcessIdFB() const
{
   std::ostringstream os;
   os << TAB << "int rval=-1;\n"
      << TAB << "if (_sendMapIter!=_sendMap.end()) rval=(*_sendMapIter).first;\n"
      << TAB << "++_sendMapIter;\n"
      << TAB << "return rval;\n";
   return os.str();
}

std::string CompCategoryBase::getAtSendProcessIdEndFB() const
{
   std::ostringstream os;
   os << TAB << "return (_sendMapIter==_sendMap.end());\n";
   return os.str();
}

std::string CompCategoryBase::getResetReceiveProcessIdIteratorsFB() const
{
   std::ostringstream os;
   os << TAB << "_demarshallerMapIter=_demarshallerMap.begin();\n";
   return os.str();
}

std::string CompCategoryBase::getGetReceiveNextProcessIdFB() const
{
   std::ostringstream os;
   os << TAB << "int rval=-1;\n"
      << TAB << "if (_demarshallerMapIter!=_demarshallerMap.end()) rval=(*_demarshallerMapIter).first;\n"
      << TAB << "++_demarshallerMapIter;\n"
      << TAB << "return rval;\n";
   return os.str();
}

std::string CompCategoryBase::getAtReceiveProcessIdEndFB() const
{
   std::ostringstream os;
   os << TAB << "return (_demarshallerMapIter==_demarshallerMap.end());\n";
   return os.str();
}

std::string CompCategoryBase::getSetMemPatternFB() const
{
   std::ostringstream os;
   os << TAB << "std::map<std::string, CG_T_GetSendTypeFunctionPtr>::iterator fiter = CG_getSendTypeTemplates.find(phaseName);\n"
      << TAB << "int nBytes=0;\n"
      << TAB << "bool inList=(fiter != CG_getSendTypeTemplates.end());\n"
      << TAB << "if (inList) {\n"
      << TAB << TAB << "ShallowArray<" << getInstanceBaseName() << "*> &nodes = _sendMap[dest];\n"
      << TAB << TAB << "inList = inList && (nodes.size()!=0);\n"
      << TAB << TAB << "if (inList) {\n"
      << TAB << TAB << TAB << "ShallowArray<" << getInstanceBaseName() << "*>::iterator niter = nodes.begin();\n"
      << TAB << TAB << TAB << "std::vector<int> npls;\n"
      << TAB << TAB << TAB << "std::vector<MPI_Aint> blocs;\n"
      << TAB << TAB << TAB << "CG_T_GetSendTypeFunctionPtr & function = (fiter->second);\n"
      << TAB << TAB << TAB << "((*niter)->*(function))(npls, blocs);\n"
      << TAB << TAB << TAB << "int npblocks=npls.size();\n"
      << TAB << TAB << TAB << "assert(npblocks==blocs.size());\n"
      << TAB << TAB << TAB << "inList = inList && (npblocks!=0);\n"
      << TAB << TAB << TAB << "if (inList) {\n"
      << TAB << TAB << TAB << TAB << "std::vector<int> nplengths;\n"
      << TAB << TAB << TAB << TAB << "std::vector<int> npdispls;\n"
      << TAB << TAB << TAB << TAB << "MPI_Aint nodeAddress;\n"
      << TAB << TAB << TAB << TAB << "MPI_Get_address(*niter, &nodeAddress);\n"
      << TAB << TAB << TAB << TAB << "mpptr->orig = reinterpret_cast<char*>(*niter);\n"
      << TAB << TAB << TAB << TAB << "nBytes += npls[0];\n"
      << TAB << TAB << TAB << TAB << "nplengths.push_back(npls[0]);\n"
      << TAB << TAB << TAB << TAB << "int dcurr, dprev=blocs[0]-nodeAddress;\n"
      << TAB << TAB << TAB << TAB << "npdispls.push_back(dprev);\n"
      << TAB << TAB << TAB << TAB << "for (int i=1; i<npblocks; ++i) {\n"
      << TAB << TAB << TAB << TAB << TAB << "nBytes += npls[i];\n"
      << TAB << TAB << TAB << TAB << TAB << "dcurr = blocs[i]-nodeAddress;;\n"
      << TAB << TAB << TAB << TAB << TAB << "if (dcurr-dprev == npls[i-1])\n"
      << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths[nplengths.size()-1] += npls[i];\n"
      << TAB << TAB << TAB << TAB << TAB << "else {\n"
      << TAB << TAB << TAB << TAB << TAB << TAB << "npdispls.push_back(dcurr);\n"
      << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths.push_back(npls[i]);\n"
      << TAB << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB << TAB << "dprev=dcurr;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB << "assert(nplengths.size()==npdispls.size());\n"
      << TAB << TAB << TAB << TAB << "int* pattern = mpptr->allocatePattern(nplengths.size());\n"
      << TAB << TAB << TAB << TAB << "std::vector<int>::iterator npliter=nplengths.begin(),\n"
      << TAB << TAB << TAB << TAB << TAB << "nplend=nplengths.end(),\n"
      << TAB << TAB << TAB << TAB << TAB << "npditer=npdispls.begin();\n"
      << TAB << TAB << TAB << TAB << "for (; npliter!=nplend; ++npliter, ++npditer, ++pattern) {\n"
      << TAB << TAB << TAB << TAB << TAB << "*pattern = *npditer;\n"
      << TAB << TAB << TAB << TAB << TAB << "*(++pattern) = *npliter;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB << "int nblocks=nodes.size();\n"
      << TAB << TAB << TAB << TAB << "nBytes *= nblocks;\n"
      << TAB << TAB << TAB << TAB << "MPI_Aint naddr, prevnaddr;\n"
      << TAB << TAB << TAB << TAB << "MPI_Get_address(*niter, &prevnaddr);\n"
      << TAB << TAB << TAB << TAB << "int* bdispls = mpptr->allocateOrigDispls(nblocks);\n"
      << TAB << TAB << TAB << TAB << "bdispls[0]=0;\n"
      << TAB << TAB << TAB << TAB << "++niter;\n"
      << TAB << TAB << TAB << TAB << "for (int i=1; i<nblocks; ++i, ++niter) {\n"
      << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(*niter, &naddr);\n"
      << TAB << TAB << TAB << TAB << TAB << "bdispls[i]=naddr-prevnaddr;\n"
      << TAB << TAB << TAB << TAB << TAB << "prevnaddr=naddr;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"
      << TAB << "return nBytes;\n";
   return os.str();
}

std::string CompCategoryBase::getGetIndexedBlockFB() const
{
   std::ostringstream os;
   os << TAB << "std::map<std::string, CG_T_GetSendTypeFunctionPtr>::iterator fiter = CG_getSendTypeTemplates.find(phaseName);\n"
      << TAB << "int nBytes=0;\n"
      << TAB << "bool inList=(fiter != CG_getSendTypeTemplates.end());\n"
      << TAB << "if (inList) {\n"
      << TAB << TAB << "ShallowArray<" << getInstanceBaseName() << "*> &nodes = _sendMap[dest];\n"
      << TAB << TAB << "inList = inList && (nodes.size()!=0);\n"
      << TAB << TAB << "if (inList) {\n"

      << TAB << TAB << TAB << "ShallowArray<" << getInstanceBaseName() << "*>::iterator niter = nodes.begin();\n"
      << TAB << TAB << TAB << "std::vector<int> npls;\n"
      << TAB << TAB << TAB << "std::vector<MPI_Aint> blocs;\n"
      << TAB << TAB << TAB << "CG_T_GetSendTypeFunctionPtr & function = (fiter->second);\n"
      << TAB << TAB << TAB << "((*niter)->*(function))(npls, blocs);\n"
      << TAB << TAB << TAB << "int npblocks=npls.size();\n"
      << TAB << TAB << TAB << "assert(npblocks==blocs.size());\n"
      << TAB << TAB << TAB << "inList = inList && (npblocks!=0);\n"
      << TAB << TAB << TAB << "if (inList) {\n"
      << TAB << TAB << TAB << TAB << "int* nplengths = new int[npblocks];\n"
      << TAB << TAB << TAB << TAB << "int* npdispls = new int[npblocks];\n"
      << TAB << TAB << TAB << TAB << "MPI_Aint nodeAddress;\n"
      << TAB << TAB << TAB << TAB << "MPI_Get_address(*niter, &nodeAddress);\n"
      << TAB << TAB << TAB << TAB << "for (int i=0; i<npblocks; ++i) {\n"
      << TAB << TAB << TAB << TAB << TAB << "nBytes += npls[i];\n"
      << TAB << TAB << TAB << TAB << TAB << "nplengths[i]=npls[i];\n"
      << TAB << TAB << TAB << TAB << TAB << "npdispls[i]=blocs[i]-nodeAddress;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB << "MPI_Datatype nodeTypeBasic, nodeType;\n"
      << TAB << TAB << TAB << TAB << "MPI_Type_indexed(npblocks, nplengths, npdispls, MPI_CHAR, &nodeTypeBasic);\n"
      << TAB << TAB << TAB << TAB << "MPI_Type_create_resized(nodeTypeBasic, 0, sizeof(" << getInstanceBaseName() << "), &nodeType);\n"
      << TAB << TAB << TAB << TAB << "delete [] nplengths;\n"
      << TAB << TAB << TAB << TAB << "delete [] npdispls;\n\n"

      << TAB << TAB << TAB << TAB << "int nblocks=nodes.size();\n"
      << TAB << TAB << TAB << TAB << "nBytes *= nblocks;\n"
      << TAB << TAB << TAB << TAB << "int* blengths = new int[nblocks];\n"
      << TAB << TAB << TAB << TAB << "MPI_Aint* bdispls = new MPI_Aint[nblocks];\n"      
      << TAB << TAB << TAB << TAB << "blockLocation=nodeAddress;\n"
      << TAB << TAB << TAB << TAB << "for (int i=0; i<nblocks; ++i, ++niter) {\n"
      << TAB << TAB << TAB << TAB << TAB << "blengths[i]=1;\n"
      << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(*niter, &bdispls[i]);\n"
      << TAB << TAB << TAB << TAB << TAB << "bdispls[i]-=blockLocation;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB << "MPI_Type_create_hindexed(nblocks, blengths, bdispls, nodeType, blockType);\n"
      << TAB << TAB << TAB << TAB << "MPI_Type_free(&nodeType);\n"
      << TAB << TAB << TAB << TAB << "delete [] blengths;\n"
      << TAB << TAB << TAB << TAB << "delete [] bdispls;\n"

      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"
      << TAB << "return nBytes;\n";
   return os.str();
}

std::string CompCategoryBase::getGetReceiveBlockCreatorFB() const
{
   std::ostringstream os;
   os << TAB << "return getDemarshaller(fromPartitionId);\n";
   return os.str();
}

std::string CompCategoryBase::getSendFB() const
{
   std::ostringstream os;
   os << TAB << TAB << TAB << "std::map<std::string, " << SENDFUNCTIONPTRTYPE 
      << ">::iterator fiter = CG_sendTemplates.find(getSimulation().getPhaseName());\n"
      << TAB << TAB << TAB << "if (fiter != CG_sendTemplates.end()) {\n"
      << TAB << TAB << TAB << TAB << "CG_T_SendFunctionPtr & function =  (fiter->second);\n"
      << TAB << TAB << TAB << TAB << "ShallowArray<" << getInstanceBaseName() <<"*> &nodes = _sendMap[pid];\n"
      << TAB << TAB << TAB << TAB << "ShallowArray<" << getInstanceBaseName() << "*>::iterator nbegin = nodes.begin(), niter, nend = nodes.end();\n"
      << TAB << TAB << TAB << TAB << "for (niter = nbegin; niter!=nend; ++niter) {\n"
      << TAB << TAB << TAB << TAB << TAB << "((*niter)->*(function))(os);\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << "}\n";
   return os.str();
}

std::string CompCategoryBase::getGetDemarshallerFB() const
{
   std::ostringstream os;
   os << TAB << "CCDemarshaller* ccd=0;\n"
      << TAB << "std::map <int, CCDemarshaller*>::iterator iter;\n"
      << TAB << "iter = _demarshallerMap.find(fromPartitionId);\n"
      << TAB << "if (iter != _demarshallerMap.end()) {\n"
      << TAB << TAB << "ccd = (*iter).second;\n"
      << TAB << "}\n"
      << TAB << "return ccd;\n";
   return os.str();
}

std::string CompCategoryBase::getFindDemarshallerFB() const
{
   std::ostringstream os;
   os << TAB << "CCDemarshaller* ccd;\n"
      << TAB << "std::map <int, CCDemarshaller*>::iterator iter;\n"
      << TAB << "iter = _demarshallerMap.find(fromPartitionId);\n"
      << TAB << "if (iter == _demarshallerMap.end()) {\n"
      << TAB << TAB << "ccd = new CCDemarshaller(&getSimulation());\n"
      << TAB << TAB << "_demarshallerMap[fromPartitionId] = ccd;\n\n"
      << TAB << TAB << "std::auto_ptr<" + getInstanceProxyDemarshallerName() + "> ap;\n"
      << TAB << TAB << getInstanceProxyName() + "::CG_recv_FLUSH_LENS_demarshaller(ap);\n"
      << TAB << TAB << "ccd->CG_recvTemplates[\"FLUSH_LENS\"] = ap.release();\n\n"
      << TAB << TAB << "std::map<std::string, Phase*>::iterator it, end = _phaseMappings.end();\n"
      << TAB << TAB << "for (it = _phaseMappings.begin(); it != end; ++it) {\n"
      << getFindDemarshallerFillerCode()
      << TAB << TAB << "}\n"
      << TAB << "} else {\n"
      << TAB << TAB << "ccd = (*iter).second;\n"
      << TAB << "}\n"
      << TAB << "return ccd;\n";
   return os.str();
}

void CompCategoryBase::addDistributionCodeToCC(Class& instance) const
{
   instance.addHeader("\"" + getInstanceProxyName() + ".h\"", MPICONDITIONAL);
//   instance.addHeader("\"" + getInstanceProxyName() + "Demarshaller.h\"", MPICONDITIONAL);
   instance.addHeader("\"" + OUTPUTSTREAM + ".h\"", MPICONDITIONAL);
   instance.addHeader("\"ShallowArray.h\"", MPICONDITIONAL);
   instance.addHeader("\"Phase.h\"", MPICONDITIONAL);
//   instance.addHeader("\"ConnectionIncrement.h\"", MPICONDITIONAL);
   instance.addHeader("\"IndexedBlockCreator.h\"", MPICONDITIONAL);
   instance.addHeader("<map>", MPICONDITIONAL);
   instance.addHeader("<list>", MPICONDITIONAL);
   instance.addHeader("<cassert>", MPICONDITIONAL);

   MacroConditional mpiConditional(MPICONDITIONAL);

   TypeDefinition sendDefinition;
   sendDefinition.setMacroConditional(mpiConditional);
   sendDefinition.setDefinition(
      "void (" + getInstanceBaseName() + "::*" + SENDFUNCTIONPTRTYPE +
      ")(" + OUTPUTSTREAM + "* ) const");
   instance.addTypeDefinition(sendDefinition);

   TypeDefinition getSendTypeDefinition;
   getSendTypeDefinition.setMacroConditional(mpiConditional);
   getSendTypeDefinition.setDefinition(
      "void (" + getInstanceBaseName() + "::*" + GETSENDTYPEFUNCTIONPTRTYPE +
      ")(std::vector<int>&, std::vector<MPI_Aint>&) const");
   instance.addTypeDefinition(getSendTypeDefinition);

   // Add addVariableNamesForPhase method
   std::auto_ptr<Method> addVariableNamesForPhaseMethod(
      new Method("addVariableNamesForPhase", "void") );
   addVariableNamesForPhaseMethod->setMacroConditional(mpiConditional);
   addVariableNamesForPhaseMethod->addParameter(
      "std::set<std::string>& " + NAMESSET);
   addVariableNamesForPhaseMethod->addParameter(
      "const std::string& " + PHASE);
   addVariableNamesForPhaseMethod->setFunctionBody(
      getAddVariableNamesForPhaseFB());
   instance.addMethod(addVariableNamesForPhaseMethod);

   // Add setDistributionTemplates method
   std::auto_ptr<Method> setDistributionTemplatesMethod(
      new Method("setDistributionTemplates", "void") );
   setDistributionTemplatesMethod->setMacroConditional(mpiConditional);
   setDistributionTemplatesMethod->setVirtual();
   setDistributionTemplatesMethod->setFunctionBody(
      getSetDistributionTemplatesFB());
   instance.addMethod(setDistributionTemplatesMethod);

   // Add resetSendProcessIdIterators method
   std::auto_ptr<Method> resetSendProcessIdIteratorsMethod(
      new Method("resetSendProcessIdIterators", "void") );
   resetSendProcessIdIteratorsMethod->setMacroConditional(mpiConditional);
   resetSendProcessIdIteratorsMethod->setVirtual();
   resetSendProcessIdIteratorsMethod->setFunctionBody(
      getResetSendProcessIdIteratorsFB());
   instance.addMethod(resetSendProcessIdIteratorsMethod);

   // Add getSendNextProcessId method
   std::auto_ptr<Method> getSendNextProcessIdMethod(
      new Method("getSendNextProcessId", "int") );
   getSendNextProcessIdMethod->setMacroConditional(mpiConditional);
   getSendNextProcessIdMethod->setVirtual();
   getSendNextProcessIdMethod->setFunctionBody(
      getGetSendNextProcessIdFB());
   instance.addMethod(getSendNextProcessIdMethod);

   // Add atSendProcessIdEnd method
   std::auto_ptr<Method> atSendProcessIdEndMethod(
      new Method("atSendProcessIdEnd", "bool") );
   atSendProcessIdEndMethod->setMacroConditional(mpiConditional);
   atSendProcessIdEndMethod->setVirtual();
   atSendProcessIdEndMethod->setFunctionBody(
      getAtSendProcessIdEndFB());
   instance.addMethod(atSendProcessIdEndMethod);

   // Add resetReceiveProcessIdIterators method
   std::auto_ptr<Method> resetReceiveProcessIdIteratorsMethod(
      new Method("resetReceiveProcessIdIterators", "void") );
   resetReceiveProcessIdIteratorsMethod->setMacroConditional(mpiConditional);
   resetReceiveProcessIdIteratorsMethod->setVirtual();
   resetReceiveProcessIdIteratorsMethod->setFunctionBody(
      getResetReceiveProcessIdIteratorsFB());
   instance.addMethod(resetReceiveProcessIdIteratorsMethod);

   // Add getReceiveNextProcessId method
   std::auto_ptr<Method> getReceiveNextProcessIdMethod(
      new Method("getReceiveNextProcessId", "int") );
   getReceiveNextProcessIdMethod->setMacroConditional(mpiConditional);
   getReceiveNextProcessIdMethod->setVirtual();
   getReceiveNextProcessIdMethod->setFunctionBody(
      getGetReceiveNextProcessIdFB());
   instance.addMethod(getReceiveNextProcessIdMethod);

   // Add atReceiveProcessIdEnd method
   std::auto_ptr<Method> atReceiveProcessIdEndMethod(
      new Method("atReceiveProcessIdEnd", "bool") );
   atReceiveProcessIdEndMethod->setMacroConditional(mpiConditional);
   atReceiveProcessIdEndMethod->setVirtual();
   atReceiveProcessIdEndMethod->setFunctionBody(
      getAtReceiveProcessIdEndFB());
   instance.addMethod(atReceiveProcessIdEndMethod);

   // Add setMemPattern method
   std::auto_ptr<Method> setMemPatternMethod(new Method("setMemPattern", "int") );
   setMemPatternMethod->setMacroConditional(mpiConditional);
   setMemPatternMethod->setVirtual();
   setMemPatternMethod->addParameter("std::string phaseName");
   setMemPatternMethod->addParameter("int dest");
   setMemPatternMethod->addParameter("MemPattern* mpptr");
   setMemPatternMethod->setFunctionBody(getSetMemPatternFB());
   instance.addMethod(setMemPatternMethod);

   // Add getIndexedBlock method
   std::auto_ptr<Method> getIndexedBlockMethod(new Method("getIndexedBlock", "int") );
   getIndexedBlockMethod->setMacroConditional(mpiConditional);
   getIndexedBlockMethod->setVirtual();
   getIndexedBlockMethod->addParameter("std::string phaseName");
   getIndexedBlockMethod->addParameter("int dest");
   getIndexedBlockMethod->addParameter("MPI_Datatype* blockType");
   getIndexedBlockMethod->addParameter("MPI_Aint& blockLocation");
   getIndexedBlockMethod->setFunctionBody(getGetIndexedBlockFB());
   instance.addMethod(getIndexedBlockMethod);

   // Add getReceiveBlockCreator method
   std::auto_ptr<Method> getReceiveBlockCreatorMethod(new Method("getReceiveBlockCreator", "IndexedBlockCreator*") );
   getReceiveBlockCreatorMethod->setMacroConditional(mpiConditional);
   getReceiveBlockCreatorMethod->setVirtual();
   getReceiveBlockCreatorMethod->addParameter("int fromPartitionId");
   getReceiveBlockCreatorMethod->setFunctionBody(getGetReceiveBlockCreatorFB());
   instance.addMethod(getReceiveBlockCreatorMethod);

   // Add send method
   std::auto_ptr<Method> sendMethod(
      new Method("send", "void") );
   sendMethod->setMacroConditional(mpiConditional);
   sendMethod->setVirtual();
   sendMethod->setInline();
   sendMethod->addParameter("int pid");
   sendMethod->addParameter("OutputStream* os");
   sendMethod->setFunctionBody(getSendFB());
   instance.addMethod(sendMethod);

   // Add getDemarshaller method
   std::auto_ptr<Method> getDemarshallerMethod(
      new Method("getDemarshaller", PREFIX+getInstanceName()+"CompCategory::CCDemarshaller*") );
   getDemarshallerMethod->setMacroConditional(mpiConditional);
   getDemarshallerMethod->setVirtual();
   getDemarshallerMethod->addParameter("int fromPartitionId");
   getDemarshallerMethod->setFunctionBody(
      getGetDemarshallerFB());
   instance.addMethod(getDemarshallerMethod);

   // Add findDemarshaller method
   std::auto_ptr<Method> findDemarshallerMethod(
      new Method("findDemarshaller", PREFIX+getInstanceName()+"CompCategory::CCDemarshaller*") );
   findDemarshallerMethod->setMacroConditional(mpiConditional);
   findDemarshallerMethod->setVirtual();
   findDemarshallerMethod->addParameter("int fromPartitionId");
   findDemarshallerMethod->setFunctionBody(
      getFindDemarshallerFB());
   instance.addMethod(findDemarshallerMethod);

   std::auto_ptr<Attribute> attCup;
   CustomAttribute* sendTemplates = new CustomAttribute(
      SENDTEMPLATES, "std::map<std::string, " + SENDFUNCTIONPTRTYPE + ">", AccessType::PRIVATE);
   sendTemplates->setMacroConditional(mpiConditional);
   attCup.reset(sendTemplates);
   instance.addAttribute(attCup);

   CustomAttribute* getSendTypeTemplates = new CustomAttribute(
      GETSENDTYPETEMPLATES, "std::map<std::string, " + GETSENDTYPEFUNCTIONPTRTYPE + ">", AccessType::PRIVATE);
   getSendTypeTemplates->setMacroConditional(mpiConditional);
   attCup.reset(getSendTypeTemplates);
   instance.addAttribute(attCup);

   /************** classes ************************/

   // Demarshaller class within CompCategory Class

   //**** member compcategory demarshaller classes
   std::auto_ptr<Class> ccdemarshaller(new Class("CCDemarshaller")); 
   std::string baseName = "Demarshaller";
   std::auto_ptr<BaseClass> demarshallerBase(new BaseClass(baseName));
   ccdemarshaller->addBaseClass(demarshallerBase);
   std::auto_ptr<BaseClass> ibcBase(new BaseClass("IndexedBlockCreator"));
   ccdemarshaller->addBaseClass(ibcBase);

   TypeDefinition recvDemarshaller;
   recvDemarshaller.setDefinition(
      "std::map<std::string, " + getInstanceProxyDemarshallerName() + "*> CG_RecvDemarshallers");
   ccdemarshaller->addTypeDefinition(recvDemarshaller);

   FriendDeclaration ccFriend(getCompCategoryBaseName());
   ccdemarshaller->addFriendDeclaration(ccFriend);

   // receiveList attribute
   CustomAttribute* receiveList = new CustomAttribute("_receiveList", "ShallowArray<"+getInstanceProxyName()+">", AccessType::PROTECTED);
   std::auto_ptr<Attribute> receiveListAp(receiveList);
   ccdemarshaller->addAttribute(receiveListAp);

   // receiveListIter attribute
   CustomAttribute* receiveListIter = new CustomAttribute("_receiveListIter", "ShallowArray<"+getInstanceProxyName()+">::iterator", AccessType::PROTECTED);
   std::auto_ptr<Attribute> receiveListIterAp(receiveListIter);
   ccdemarshaller->addAttribute(receiveListIterAp);

   // receiveState attribute
   CustomAttribute* receiveState = new CustomAttribute("_receiveState", "ShallowArray<"+getInstanceProxyName()+">::iterator", AccessType::PROTECTED);
   std::auto_ptr<Attribute> receiveStateAp(receiveState);
   ccdemarshaller->addAttribute(receiveStateAp);

   // recvTemplates attribute
   CustomAttribute* recvTemplates = new CustomAttribute("CG_recvTemplates", "CG_RecvDemarshallers", AccessType::PROTECTED);
   std::auto_ptr<Attribute> recvTemplatesAp(recvTemplates);
   ccdemarshaller->addAttribute(recvTemplatesAp);

   // simulation pointer attribute
   CustomAttribute* simulationPtr = new CustomAttribute("_sim", "Simulation", AccessType::PRIVATE);
   simulationPtr->setPointer();
   std::auto_ptr<Attribute> simulationPtrAp(simulationPtr);
   ccdemarshaller->addAttribute(simulationPtrAp);

   // Constructors
   std::auto_ptr<ConstructorMethod> constructor(new ConstructorMethod("CCDemarshaller"));

   std::ostringstream constructorFB;
   constructorFB << TAB << TAB << TAB << "_sim = sim;\n";
   std::ostringstream initString;
   std::ostringstream setMemPatternMethodFB;
   std::ostringstream getIndexedBlockMethodFB;
   std::ostringstream demarshallMethodFB;
   std::ostringstream addDestinationMethodFB;
   std::ostringstream resetMethodFB;
   std::ostringstream doneMethodFB;
   initString << (baseName + "()");

   setMemPatternMethodFB << TAB << TAB << TAB << "CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(phaseName);\n"
       << TAB << TAB << TAB << "int nBytes=0;\n"
       << TAB << TAB << TAB << "bool inList = (diter != CG_recvTemplates.end());\n"
       << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << "inList = inList && (_receiveList.size()!=0);\n"
       << TAB << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << TAB << "ShallowArray<" << getInstanceProxyName() << ">::iterator niter=_receiveList.begin();\n"
       << TAB << TAB << TAB << TAB << TAB << getInstanceProxyDemarshallerName() << "* dm = diter->second;\n"
       << TAB << TAB << TAB << TAB << TAB << "std::vector<MPI_Aint> blocs;\n"
       << TAB << TAB << TAB << TAB << TAB << "std::vector<int> npls;\n"
       << TAB << TAB << TAB << TAB << TAB << "dm->setDestination(&(*niter));\n"
       << TAB << TAB << TAB << TAB << TAB << "dm->getBlocks(npls, blocs);\n"
       << TAB << TAB << TAB << TAB << TAB << "int npblocks=npls.size();\n"
       << TAB << TAB << TAB << TAB << TAB << "assert(npblocks==blocs.size());\n"
       << TAB << TAB << TAB << TAB << TAB << "inList = inList && (npblocks!=0);\n"
       << TAB << TAB << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "std::vector<int> nplengths;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "std::vector<int> npdispls;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Aint nodeAddress;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(&(*niter), &nodeAddress);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "mpptr->orig = reinterpret_cast<char*>(&(*niter));\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "nBytes += npls[0];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths.push_back(npls[0]);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int dcurr, dprev=blocs[0]-nodeAddress;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "npdispls.push_back(dprev);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "for (int i=1; i<npblocks; ++i) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nBytes += npls[i];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "dcurr = blocs[i]-nodeAddress;;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "if (dcurr-dprev == npls[i-1])\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths[nplengths.size()-1] += npls[i];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "else {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "npdispls.push_back(dcurr);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths.push_back(npls[i]);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "dprev=dcurr;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "assert(nplengths.size()==npdispls.size());\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int* pattern = mpptr->allocatePattern(nplengths.size());\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "std::vector<int>::iterator npliter=nplengths.begin(),\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nplend=nplengths.end(),\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "npditer=npdispls.begin();\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "for (; npliter!=nplend; ++npliter, ++npditer, ++pattern) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "*pattern = *npditer;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "*(++pattern) = *npliter;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int nblocks=_receiveList.size();\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "nBytes *= nblocks;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Aint naddr, prevnaddr;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(&(*niter), &prevnaddr);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int* bdispls = mpptr->allocateOrigDispls(nblocks);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "bdispls[0]=0;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "++niter;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "for (int i=1; i<nblocks; ++i, ++niter) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(&(*niter), &naddr);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "bdispls[i]=naddr-prevnaddr;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "prevnaddr=naddr;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "return nBytes;\n";

   getIndexedBlockMethodFB << TAB << TAB << TAB << "CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(phaseName);\n"
       << TAB << TAB << TAB << "int nBytes=0;\n"
       << TAB << TAB << TAB << "bool inList = (diter != CG_recvTemplates.end());\n"
       << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << "inList = inList && (_receiveList.size()!=0);\n"
       << TAB << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << TAB << "ShallowArray<" << getInstanceProxyName() << ">::iterator niter=_receiveList.begin();\n"
       << TAB << TAB << TAB << TAB << TAB << getInstanceProxyDemarshallerName() << "* dm = diter->second;\n"
       << TAB << TAB << TAB << TAB << TAB << "std::vector<MPI_Aint> blocs;\n"
       << TAB << TAB << TAB << TAB << TAB << "std::vector<int> npls;\n"
       << TAB << TAB << TAB << TAB << TAB << "dm->setDestination(&(*niter));\n"
       << TAB << TAB << TAB << TAB << TAB << "dm->getBlocks(npls, blocs);\n"
       << TAB << TAB << TAB << TAB << TAB << "int npblocks=npls.size();\n"
       << TAB << TAB << TAB << TAB << TAB << "assert(npblocks==blocs.size());\n"
       << TAB << TAB << TAB << TAB << TAB << "inList = inList && (npblocks!=0);\n"
       << TAB << TAB << TAB << TAB << TAB << "if (inList) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int* nplengths = new int[npblocks];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int* npdispls = new int[npblocks];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Aint nodeAddress;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(&(*niter), &nodeAddress);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "for (int i=0; i<npblocks; ++i) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nBytes += npls[i];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "nplengths[i]=npls[i];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "npdispls[i]=blocs[i]-nodeAddress;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Datatype proxyTypeBasic, proxyType;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Type_indexed(npblocks, nplengths, npdispls, MPI_CHAR, &proxyTypeBasic);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Type_create_resized(proxyTypeBasic, 0, sizeof(" << getInstanceProxyName() 
			                  << "), &proxyType);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "delete [] nplengths;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "delete [] npdispls;\n\n"

       << TAB << TAB << TAB << TAB << TAB << TAB << "int nblocks=_receiveList.size();\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "nBytes *= nblocks;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "int* blengths = new int[nblocks];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Aint* bdispls = new MPI_Aint[nblocks];\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "blockLocation=nodeAddress;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "for (int i=0; i<nblocks; ++i, ++niter) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "blengths[i]=1;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Get_address(&(*niter), &bdispls[i]);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "bdispls[i]-=blockLocation;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Type_create_hindexed(nblocks, blengths, bdispls, proxyType, blockType);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "MPI_Type_free(&proxyType);\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "delete [] blengths;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "delete [] bdispls;\n"

       << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "return nBytes;\n";

   demarshallMethodFB << TAB << TAB << TAB << "int buffSize = size;\n\n"
       << TAB << TAB << TAB << "CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());\n"
       << TAB << TAB << TAB << "if (diter != CG_recvTemplates.end()) {\n"
		      << TAB << TAB << TAB << TAB << getInstanceProxyDemarshallerName() + "* dm = diter->second;\n"
       << TAB << TAB << TAB << TAB << "ShallowArray<"+getInstanceProxyName()+">::iterator &niter = _receiveState;\n"
       << TAB << TAB << TAB << TAB << "ShallowArray<"+getInstanceProxyName()+">::iterator nend = _receiveList.end();\n"
       << TAB << TAB << TAB << TAB << "const char* buff = buffer;\n"
       << TAB << TAB << TAB << TAB << "while (niter!=nend && buffSize!=0) {\n"
       << TAB << TAB << TAB << TAB << TAB << "buffSize = dm->demarshall(buff, buffSize);\n"
       << TAB << TAB << TAB << TAB << TAB << "buff = buffer+(size-buffSize);\n"
       << TAB << TAB << TAB << TAB << TAB << "if (buffSize!=0 || dm->done()) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "++niter;\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "if (niter != nend) {\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << TAB << "dm->setDestination(&(*niter));\n"
       << TAB << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "}\n"
       << TAB << TAB << TAB << "return buffSize;\n";

   addDestinationMethodFB << TAB << TAB << TAB << "_receiveList.increaseSizeTo(_receiveList.size()+1);\n"
			  << TAB << TAB << TAB << "return &_receiveList[_receiveList.size()-1];\n";

   resetMethodFB << TAB << TAB << TAB << "CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());\n"
       << TAB << TAB << TAB << "if (diter != CG_recvTemplates.end()) {\n"
       << TAB << TAB << TAB << TAB << "_receiveState= _receiveList.begin();\n"
       << TAB << TAB << TAB << TAB << "CG_recvTemplates[_sim->getPhaseName()]->setDestination(&(*_receiveState));\n"
       << TAB << TAB << TAB << "}\n";

   doneMethodFB << TAB << TAB << TAB << "CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());\n"
       << TAB << TAB << TAB << "return (diter == CG_recvTemplates.end() || _receiveState == _receiveList.end());\n";

   // add constructor
   constructor->addParameter("Simulation* sim");
   constructor->setInitializationStr(initString.str());
   constructor->setFunctionBody(constructorFB.str());
   constructor->setInline();

   std::auto_ptr<Method> consToIns1(constructor.release());
   ccdemarshaller->addMethod(consToIns1);
   
   // add setMemPattern method
   std::auto_ptr<Method> demarshallerSetMemPatternMethod(new Method("setMemPattern", "int"));
   demarshallerSetMemPatternMethod->setInline();
   demarshallerSetMemPatternMethod->addParameter("std::string phaseName");
   demarshallerSetMemPatternMethod->addParameter("int source");
   demarshallerSetMemPatternMethod->addParameter("MemPattern* mpptr");
   demarshallerSetMemPatternMethod->setFunctionBody(setMemPatternMethodFB.str());
   ccdemarshaller->addMethod(demarshallerSetMemPatternMethod);

   // add getIndexedBlock method
   std::auto_ptr<Method> demarshallerGetIndexedBlockMethod(new Method("getIndexedBlock", "int"));
   demarshallerGetIndexedBlockMethod->setInline();
   demarshallerGetIndexedBlockMethod->addParameter("std::string phaseName");
   demarshallerGetIndexedBlockMethod->addParameter("int source");
   demarshallerGetIndexedBlockMethod->addParameter("MPI_Datatype* blockType");
   demarshallerGetIndexedBlockMethod->addParameter("MPI_Aint& blockLocation");
   demarshallerGetIndexedBlockMethod->setFunctionBody(getIndexedBlockMethodFB.str());
   ccdemarshaller->addMethod(demarshallerGetIndexedBlockMethod);

   // add demarshall method
   std::auto_ptr<Method> demarshallMethod(new Method("demarshall", "int"));
   demarshallMethod->setInline();
   demarshallMethod->addParameter("const char* buffer");
   demarshallMethod->addParameter("int size");
   demarshallMethod->setFunctionBody(demarshallMethodFB.str());
   ccdemarshaller->addMethod(demarshallMethod);

   // add addDestination method
   std::auto_ptr<Method> addDestinationMethod(new Method("addDestination", getType() + "ProxyBase*"));
   addDestinationMethod->setInline();
   addDestinationMethod->setFunctionBody(addDestinationMethodFB.str());
   ccdemarshaller->addMethod(addDestinationMethod);

   // add reset method
   std::auto_ptr<Method> resetMethod(new Method("reset", "void"));
   resetMethod->setInline();
   resetMethod->setFunctionBody(resetMethodFB.str());
   ccdemarshaller->addMethod(resetMethod);

   // add done method
   std::auto_ptr<Method> doneMethod(new Method("done", "bool"));
   doneMethod->setInline();
   doneMethod->setFunctionBody(doneMethodFB.str());
   ccdemarshaller->addMethod(doneMethod);

   ccdemarshaller->addBasicInlineDestructor();
   ccdemarshaller->setMacroConditional(mpiConditional);

   instance.addMemberClass(ccdemarshaller, AccessType::PRIVATE);
}

std::string CompCategoryBase::getTemplateFillerCode() const
{
  std::ostringstream os;
  if (_instancePhases) {
     std::vector<Phase*>::const_iterator it, end = _instancePhases->end();
     for (it = _instancePhases->begin(); it != end; ++it) {
       if ((*it)->hasPackedVariables()) {
         os << TAB << TAB << "if (it->second->getName() == getSimulationPhaseName(\"" << (*it)->getName() << "\")){\n"
	    << TAB << TAB << TAB << SENDTEMPLATES <<"[it->second->getName()] = &" << getInstanceBaseName() 
   	 << "::CG_send_" << (*it)->getName() << ";\n"
	    << TAB << TAB << TAB << GETSENDTYPETEMPLATES << "[it->second->getName()] = &" << getInstanceBaseName() 
   	 << "::CG_getSendType_" << (*it)->getName() << ";\n"
   	 << TAB << TAB << "}\n";
       }
     }
  }
  return os.str();
}

std::string CompCategoryBase::getFindDemarshallerFillerCode() const
{
  std::ostringstream os;
  if (_instancePhases) {
     std::vector<Phase*>::const_iterator it, end = _instancePhases->end();
     for (it = _instancePhases->begin(); it != end; ++it) {
       if ((*it)->hasPackedVariables()) {
         os << TAB << TAB << TAB << "if (it->second->getName() == getSimulationPhaseName(\"" << (*it)->getName() << "\")){\n"
	    << TAB << TAB << TAB << TAB << "std::auto_ptr<" << getInstanceProxyDemarshallerName() << "> ap;\n"
   	 << TAB << TAB << TAB << TAB << getInstanceProxyName() << "::CG_recv_" << (*it)->getName() << "_demarshaller(ap);\n"
   	 << TAB << TAB << TAB << TAB << "ccd->CG_recvTemplates[(it->second->getName())] = ap.release();\n"
   	 << TAB << TAB << TAB << "}\n";
       }
     }
  }
  return os.str();
}
