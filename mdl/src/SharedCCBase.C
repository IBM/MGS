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

#include "SharedCCBase.h"
#include "ConnectionCCBase.h"
#include "Generatable.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "DuplicateException.h"
#include "SyntaxErrorException.h"
#include "FinalPhase.h"
#include "InitPhase.h"
#include "LoadPhase.h"
#include "RuntimePhase.h"
#include "Constants.h"
#include "CustomAttribute.h"
#include "Attribute.h"
#include "ConstructorMethod.h"
#include "Method.h"
#include "AccessType.h"
#include "BaseClass.h"
#include "FriendDeclaration.h"
#include "Utility.h"
#include <memory>
#include <cassert>
#include <sstream>

SharedCCBase::SharedCCBase(const std::string& fileName)
   : ConnectionCCBase(fileName)
{
}

SharedCCBase::SharedCCBase(const SharedCCBase& rv)
   : ConnectionCCBase(rv)
{
   copyOwnedHeap(rv);
}

SharedCCBase& SharedCCBase::operator=(const SharedCCBase& rv)
{
   if (this != &rv) {
      ConnectionCCBase::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void SharedCCBase::copyOwnedHeap(const SharedCCBase& rv)
{
   if (rv._sharedPhases.size() > 0) {
      std::vector<Phase*>::const_iterator it, end = rv._sharedPhases.end();
      std::auto_ptr<Phase> dup;   
      for (it = rv._sharedPhases.begin(); it != end; ++it) {
	 (*it)->duplicate(dup);
	 _sharedPhases.push_back(dup.release());
      }
   } 
   if (rv._sharedTriggeredFunctions.size() > 0) {
      std::vector<TriggeredFunction*>::const_iterator it, 
	 end = rv._sharedTriggeredFunctions.end();
      std::auto_ptr<TriggeredFunction> dup;   
      for (it = rv._sharedTriggeredFunctions.begin(); it != end; ++it) {
	 (*it)->duplicate(dup);
	 _sharedTriggeredFunctions.push_back(dup.release());
      }
   } 
}

void SharedCCBase::destructOwnedHeap()
{
   if (_sharedPhases.size() > 0) {
      std::vector<Phase*>::iterator it, end = _sharedPhases.end();
      for (it = _sharedPhases.begin(); it != end; ++it) {
	 delete *it;
      }
      _sharedPhases.clear();
   }
   if (_sharedTriggeredFunctions.size() > 0) {
      std::vector<TriggeredFunction*>::iterator it, 
	 end = _sharedTriggeredFunctions.end();
      for (it = _sharedTriggeredFunctions.begin(); it != end; ++it) {
	 delete *it;
      }
      _sharedTriggeredFunctions.clear();
   }
}

void SharedCCBase::addSharedPhase(std::auto_ptr<Phase>& phase)
{
   // For instance phases.
   isDuplicatePhase(phase.get());

   // For Shared[current] phases. 
   if (mdl::findInPhases(phase->getName(), _sharedPhases)) {
      std::ostringstream os;
      os << "phase " << phase->getName() << " is already included as an " 
	 << phase->getType();
      throw DuplicateException(os.str());
   }

   _sharedPhases.push_back(phase.release());
}

void SharedCCBase::addSharedTriggeredFunction(
   std::auto_ptr<TriggeredFunction>& triggeredFunction)
{
   // For instance triggered functions.
   isDuplicateTriggeredFunction(triggeredFunction->getName());

   // For Shared[current] triggered functions. 
   if (mdl::findInTriggeredFunctions(triggeredFunction->getName(), 
			 _sharedTriggeredFunctions)) {
      std::ostringstream os;
      os << "Triggered function " << triggeredFunction->getName() 
	 << " is already included";
      throw DuplicateException(os.str());
   }

   _sharedTriggeredFunctions.push_back(triggeredFunction.release());
}

std::string SharedCCBase::generateExtra() const
{
   std::ostringstream os;
   os << ConnectionCCBase::generateExtra();
   os << "\tShared {\n";
   if (_shareds.size() > 0) {
      MemberContainer<DataType>::const_iterator it, end = _shareds.end();
      for (it = _shareds.begin(); it != end; ++it) {
	 os  << "\t\t" << it->second->getString() << ";\n";
      }
   }
   if (_optionalSharedServices.size() > 0) {
      MemberContainer<DataType>::const_iterator it, 
	 end = _optionalSharedServices.end();
      for (it = _optionalSharedServices.begin(); it != end; ++it) {
	 os  << "\t\tOptional " << it->second->getString() << ";\n";
      }
   }
   if ((_sharedPhases.size() > 0) && (_shareds.size() > 0) && 
       (_sharedTriggeredFunctions.size() > 0)) {
      os << "\n";
   }
   if (_sharedPhases.size() > 0) {
      std::vector<Phase*>::const_iterator it, end = _sharedPhases.end();
      for (it = _sharedPhases.begin(); it != end; ++it) {
	 os << "\t\t" << (*it)->getGenerateString() << ";\n";
      }
   }
   if ((_shareds.size() > 0) && (_sharedTriggeredFunctions.size() > 0)) {
      os << "\n";
   }
   if (_sharedTriggeredFunctions.size() > 0) {
      std::vector<TriggeredFunction*>::const_iterator it, 
	 end = _sharedTriggeredFunctions.end();
      for (it = _sharedTriggeredFunctions.begin(); it != end; ++it) {
	 os << (*it)->getString() << "\n";
      }
   }
   os << "\t}\n";
   return os.str();
}

SharedCCBase::~SharedCCBase() 
{
   destructOwnedHeap();
}

void SharedCCBase::addExtraServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
   instance->addDataTypeHeaders(_shareds);
}

void SharedCCBase::addExtraOptionalServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
   instance->addDataTypeHeaders(_optionalSharedServices);
}

std::string SharedCCBase::getExtraServices(const std::string& tab) const
{
   return createServices(_shareds, tab);
}

std::string SharedCCBase::getExtraOptionalServices(
   const std::string& tab) const
{
   return createOptionalServices(_optionalSharedServices, tab);
}

std::string SharedCCBase::getExtraServiceNames(const std::string& tab) const
{
   return createServiceNames(_shareds, tab);
}

std::string SharedCCBase::getExtraOptionalServiceNames(
   const std::string& tab) const
{
   return createOptionalServiceNames(_optionalSharedServices, tab);
}

std::string SharedCCBase::getExtraServiceDescriptions(
   const std::string& tab) const
{
   return createServiceDescriptions(_shareds, tab);
}

std::string SharedCCBase::getExtraOptionalServiceDescriptions(
   const std::string& tab) const
{
   return createOptionalServiceDescriptions(_optionalSharedServices, tab);
}

std::string SharedCCBase::getExtraServiceDescriptors(
   const std::string& tab) const
{
   return createServiceDescriptors(_shareds, tab);
}

std::string SharedCCBase::getExtraOptionalServiceDescriptors(
   const std::string& tab) const
{
   return createOptionalServiceDescriptors(_optionalSharedServices, tab);
}

std::string SharedCCBase::createGetWorkUnitsMethodBody(
   const std::string& phaseName, const std::string& workUnits) const
{
   std::ostringstream os;
   os << CompCategoryBase::createGetWorkUnitsMethodBody(phaseName, workUnits);
   if (_sharedPhases.size() > 0) {
      os << createGetWorkUnitsMethodCommonBody(
	 phaseName, workUnits, getInstanceName(), _sharedPhases);
   }
   return os.str();
}

void SharedCCBase::addExtraInstanceBaseMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceBaseMethods(instance);

   instance.addHeader("\"" + getSharedMembersName() + ".h\"");

   // Add the friends
   FriendDeclaration compCategoryFriend(getCompCategoryName());
   instance.addFriendDeclaration(compCategoryFriend);

   // getSharedMembers method
   std::auto_ptr<Method> getSharedMembersMethod(
      new Method("getSharedMembers", "const " + getSharedMembersName() + "&"));
   getSharedMembersMethod->setConst();
   getSharedMembersMethod->setFunctionBody(
      TAB + "return *" + getCompCategoryBaseName() + "::"
      + getSharedMembersAttName() + ";\n");
   instance.addMethod(getSharedMembersMethod);

   // getNonConstSharedMembers method
   std::auto_ptr<Method> getNonConstSharedMembersMethod(
      new Method("getNonConstSharedMembers", getSharedMembersName() + "&"));
   getNonConstSharedMembersMethod->setAccessType(AccessType::PRIVATE);
   getNonConstSharedMembersMethod->setFunctionBody(
      TAB + "return *" + getCompCategoryBaseName() + "::"
      + getSharedMembersAttName() + ";\n");
   instance.addMethod(getNonConstSharedMembersMethod);
}

void SharedCCBase::addExtraInstanceProxyMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceProxyMethods(instance);

   instance.addHeader("\"" + getSharedMembersName() + ".h\"");

   // Add the friends
   FriendDeclaration compCategoryFriend(getCompCategoryName());
   instance.addFriendDeclaration(compCategoryFriend);

   // getSharedMembers method
   std::auto_ptr<Method> getSharedMembersMethod(
      new Method("getSharedMembers", "const " + getSharedMembersName() + "&"));
   getSharedMembersMethod->setFunctionBody(
      TAB + "return *" + getCompCategoryBaseName() + "::"
      + getSharedMembersAttName() + ";\n");
   instance.addMethod(getSharedMembersMethod);

   // getNonConstSharedMembers method
   std::auto_ptr<Method> getNonConstSharedMembersMethod(
      new Method("getNonConstSharedMembers", getSharedMembersName() + "&"));
   getNonConstSharedMembersMethod->setAccessType(AccessType::PRIVATE);
   getNonConstSharedMembersMethod->setFunctionBody(
      TAB + "return *" + getCompCategoryBaseName() + "::"
      + getSharedMembersAttName() + ";\n");
   instance.addMethod(getNonConstSharedMembersMethod);
}

void SharedCCBase::addExtraInstanceMethods(Class& instance) const
{
   ConnectionCCBase::addExtraInstanceMethods(instance);
}

void SharedCCBase::addExtraCompCategoryBaseMethods(Class& instance) const
{
   // Add the phase methods
   std::vector<Phase*>::const_iterator it, end = _sharedPhases.end();
   for (it = _sharedPhases.begin(); it != end; ++it) {
      (*it)->generateVirtualUserMethod(instance);
   }

   instance.addHeader("\"" + getSharedMembersName() + ".h\"");
   instance.addHeader("\"" + getWorkUnitSharedName() + ".h\"");
   instance.addExtraSourceHeader("\"" + getCompCategoryName() + ".h\"");

   std::auto_ptr<Attribute> attCup;
   CustomAttribute* shared = new CustomAttribute(getSharedMembersAttName(), 
						 getSharedMembersName(), 
						 AccessType::PUBLIC);
   shared->setStatic();
   shared->setPointer();
   attCup.reset(shared);
   instance.addAttribute(attCup);

   // getSharedMembers method
   std::auto_ptr<Method> getSharedMembersMethod(
      new Method("getSharedMembers", getSharedMembersName() + "&"));
   getSharedMembersMethod->setFunctionBody(
      TAB + "return *" + getSharedMembersAttName() + ";\n");
   instance.addMethod(getSharedMembersMethod);
}

void SharedCCBase::addExtraCompCategoryMethods(Class& instance) const
{
   // Add the phase methods
   std::vector<Phase*>::const_iterator it, end = _sharedPhases.end();
   for (it = _sharedPhases.begin(); it != end; ++it) {
      (*it)->generateUserMethod(instance);
   }
}

void SharedCCBase::generateSharedMembers()
{
   std::auto_ptr<Class> instance(new Class(getSharedMembersName()));

   instance->addHeader("<memory>");
   instance->addHeader("<sstream>");
   instance->addExtraSourceHeader("\"SyntaxErrorException.h\"");
   instance->addClass("NDPairList");

   addExtraOptionalServiceHeaders(instance);

   // Add the instances.
   instance->addAttributes(_shareds, AccessType::PUBLIC);

   std::auto_ptr<Method> setUpMethod(new Method("setUp", "void") );
   setUpMethod->addParameter("const NDPairList& ndplist");
   setUpMethod->setVirtual();
   setUpMethod->setFunctionBody(getSetupFromNDPairListMethodBody(_shareds));
   instance->addMethod(setUpMethod);

   mdl::addOptionalServicesToClass(*(instance.get()), 
				   _optionalSharedServices);

   instance->addStandardMethods();
   _classes.push_back(instance.release());  
}

std::string SharedCCBase::getCompCategoryBaseConstructorBody() const
{
   std::ostringstream os;
   os << TAB << "if (" << PREFIX << "sharedMembers==0) {\n";
   os << TAB << TAB << PREFIX << "sharedMembers = new " << PREFIX << getInstanceName() << "SharedMembers();\n";
   os << TAB << TAB << PREFIX << "sharedMembers->setUp(ndpList);\n";
   os << TAB << "}\n";
   os << CompCategoryBase::getCompCategoryBaseConstructorBody();
   if (_sharedPhases.size() > 0) {
      std::vector<Phase*>::const_iterator it, end = _sharedPhases.end();
      for (it = _sharedPhases.begin(); it != end; ++it) {
	 os << (*it)->getInitializePhaseMethodBody();
      }
   }
   return os.str();
}

void SharedCCBase::generateWorkUnitShared()
{
   std::string compCatName = getCompCategoryName();
   std::string fullName = getWorkUnitCommonName("Shared");
   std::auto_ptr<Class> instance(new Class(fullName));

   std::string baseName = "WorkUnit";
   std::auto_ptr<BaseClass> base(
      new BaseClass(baseName));
   instance->addBaseClass(base);

   std::string computeStateAttName = 
      "(" + compCatName + "::*_computeState) (RNG&)"; 

   std::string computeStatePar = 
      "(" + compCatName + "::*computeState) (RNG&)"; 

   instance->addHeader("\"" + baseName + ".h\"");
   instance->addClass(compCatName);
   instance->addClass(getCompCategoryBaseName());
   instance->addHeader("\"rndm.h\"");
   
   std::auto_ptr<Attribute> attCup;

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
   constructor->addParameter("void " + computeStatePar);
   constructor->addParameter(getCompCategoryBaseName() + "* compCategory");
   std::string initializationStr = "WorkUnit(), ";
   initializationStr  += "_computeState(computeState)";
   constructor->setInitializationStr(initializationStr);
   std::ostringstream constructorFB;
   constructorFB 
      << TAB << "_compCategory = static_cast<" 
      << compCatName << "*>(compCategory);\n"
      << TAB << "_rng.reSeedShared(urandom(_compCategory->getSimulation().getSharedWorkUnitRandomSeedGenerator()));\n";   
   constructor->setFunctionBody(constructorFB.str());
   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);

   // execute 
   std::auto_ptr<Method> executeMethod(
      new Method("execute", "void"));
   executeMethod->setVirtual();
   std::ostringstream executeMethodFB;   
   executeMethodFB
      << TAB << "(*_compCategory.*_computeState)(";
   executeMethodFB
      << "_rng";
   executeMethodFB
      << ");\n";
   executeMethod->setFunctionBody(executeMethodFB.str());
   instance->addMethod(executeMethod);

   // Don't add the standard methods
   instance->addBasicDestructor();
   _classes.push_back(instance.release());  


}

std::string SharedCCBase::getLoadedInstanceTypeName()
{
   return getCompCategoryName();
}

std::string SharedCCBase::getLoadedInstanceTypeArguments()
{
   return "s, \"" + getModuleName() + "\", ndpList";
}

void SharedCCBase::checkInstanceVariableNameSpace(const std::string& name) const 
{
   InterfaceImplementorBase::checkInstanceVariableNameSpace(name);
   if (_shareds.exists(name)) {
      std::ostringstream os;
      os << name + " is already included as an shared variable.";
      throw SyntaxErrorException(os.str()); 
   } else if (_optionalSharedServices.exists(name)) {
      std::ostringstream os;
      os << name + " is already included as an optional shared service.";
      throw SyntaxErrorException(os.str()); 
   }
}
