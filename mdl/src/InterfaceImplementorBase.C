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

#include "InterfaceImplementorBase.h"
#include "Generatable.h"
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
#include "FriendDeclaration.h"
#include "NotFoundException.h"
#include "SyntaxErrorException.h"
#include "MacroConditional.h"
#include <memory>
#include <sstream>
#include <iostream>
#include <cassert>
#include <list>
#include <stdio.h>
#include <string.h>
#include <algorithm>

InterfaceImplementorBase::InterfaceImplementorBase(
   const std::string& fileName) 
   : Generatable(fileName),  _instancePhases(0), _name(""), _outAttrPSet(0)
{
}

InterfaceImplementorBase::InterfaceImplementorBase(
   const InterfaceImplementorBase& rv)
   : Generatable(rv), _instances(rv._instances), 
     _interfaces(rv._interfaces), _interfaceImplementors(rv._interfaceImplementors),  
     _name(rv._name), _optionalInstanceServices(rv._optionalInstanceServices)
{
   copyOwnedHeap(rv);
}

InterfaceImplementorBase& InterfaceImplementorBase::operator=(
   const InterfaceImplementorBase& rv)
{
   if (this != &rv) {
      Generatable::operator=(rv);
      _instances = rv._instances;
      _interfaces = rv._interfaces;
      _name = rv._name;
      _interfaceImplementors = rv._interfaceImplementors;
      _optionalInstanceServices = rv._optionalInstanceServices;
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void InterfaceImplementorBase::releaseOutAttrPSet(
   std::auto_ptr<StructType>& oap)
{
   oap.reset(_outAttrPSet);
   _outAttrPSet = 0;
}

void InterfaceImplementorBase::setOutAttrPSet(std::auto_ptr<StructType>& oap)
{
   delete _outAttrPSet;
   _outAttrPSet = oap.release();
}

void InterfaceImplementorBase::setInstancePhases(
   std::auto_ptr<std::vector<Phase*> >& phases)
{
   delete _instancePhases;
   _instancePhases = phases.release();
}

void InterfaceImplementorBase::generate() const
{
   std::ostringstream os;
   os << getType() << " " << _name;
   if (_interfaces.size() != 0) {
      os << " " << "Implements ";
      MemberContainer<MemberToInterface>::const_iterator it, next, 
	 end = _interfaces.end();
      for (it = _interfaces.begin(); it != end; it++) {
	 os << it->first;
	 next = it;
	 next++;
	 if ( next != end) {
	    os << ",";
	 }
	 os << " ";
      }
   }
   os << "{\n";
   if (_instances.size() > 0) {
      MemberContainer<DataType>::const_iterator it, end = _instances.end();
      for (it = _instances.begin(); it != end; it ++) {
	 os  << "\t" << it->second->getString() << ";\n";
      }
   }
   if (_optionalInstanceServices.size() > 0) {
      MemberContainer<DataType>::const_iterator it, 
	 end = _optionalInstanceServices.end();
      for (it = _optionalInstanceServices.begin(); it != end; it ++) {
	 os  << "\tOptional " << it->second->getString() << ";\n";
      }
   }
   MemberContainer<MemberToInterface>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; ++it) {
      os << it->second->getMemberToInterfaceString(it->first);
   }
   os << _outAttrPSet->getOutAttrPSetStr();  
   os << generateExtra();
   os << "}\n\n";
   std::cout << os.str();
}

std::string InterfaceImplementorBase::generateExtra() const
{
   return "";
}

void InterfaceImplementorBase::checkAllMemberToInterfaces() 
{
   MemberContainer<MemberToInterface>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; it++)  {
      if (it->second->checkAllMapped() == false) {
	 std::cerr 
	    << "In " << _name << ", all members of interface " << it->first 
	    << " aren't mapped to a member of " << _name << "." << std::endl;
	 throw GeneralException("");
      }
   }
}

InterfaceImplementorBase::~InterfaceImplementorBase() {
   destructOwnedHeap();
}

void InterfaceImplementorBase::copyOwnedHeap(
   const InterfaceImplementorBase& rv)
{
   if (rv._outAttrPSet) {
      std::auto_ptr<StructType> dup;
      rv._outAttrPSet->duplicate(dup);
      _outAttrPSet = dup.release();
   } else {
      _outAttrPSet = 0;
   }
   if (_instancePhases) _instancePhases = new std::vector<Phase*>(*(rv._instancePhases));
   else _instancePhases = 0;
}

void InterfaceImplementorBase::destructOwnedHeap()
{
   delete _outAttrPSet;
}


std::string InterfaceImplementorBase::getModuleName() const
{
   return getName();
}

void InterfaceImplementorBase::createPSetClass(
   std::auto_ptr<Class>& instance,
   const MemberContainer<DataType>& members,
   const std::string& name) const 
{
   std::string fullName = getCommonPSetName(name);
   instance.reset(new Class(fullName));
   
   std::auto_ptr<BaseClass> psetBase(new BaseClass("ParameterSet"));
   
   instance->addBaseClass(psetBase);
   instance->addHeader("\"ParameterSet.h\"");
   instance->addHeader("\"NDPair.h\"");
   instance->addHeader("\"NDPairList.h\"");
   instance->addHeader("\"SyntaxErrorException.h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<typeinfo>");
   instance->addExtraSourceHeader("<sstream>");
   instance->addAttributes(members);

   std::auto_ptr<Method> setUpMethod(new Method("set", "void") );
   setUpMethod->addParameter("NDPairList& ndplist");
   setUpMethod->setVirtual();
   // setUpMethod->setAccessType(AccessType::PROTECTED);
   setUpMethod->setFunctionBody(getSetupFromNDPairListMethodBody(members));
   instance->addMethod(setUpMethod);

   instance->addStandardMethods();
}

void InterfaceImplementorBase::generateOutAttrPSet()
{
   std::auto_ptr<Class> instance;
   createPSetClass(instance, _outAttrPSet->_members, "OutAttr");
   _classes.push_back(instance.release());
}

void InterfaceImplementorBase::generatePublisher()
{
   std::auto_ptr<Class> instance(new Class(getPublisherName()));
   
   std::auto_ptr<Attribute> attCup;

   std::auto_ptr<BaseClass> base(
      new BaseClass("GeneratedPublisherBase"));
   
   CustomAttribute* sim = new CustomAttribute("_sim", "Simulation");
   sim->setReference();
   attCup.reset(sim);
   base->addAttribute(attCup);

   instance->addBaseClass(base);
   instance->addHeader("\"GeneratedPublisherBase.h\"");
   instance->addHeader("<memory>");
   instance->addHeader("\"Simulation.h\"");
   instance->addHeader("\"Publishable.h\"");
   instance->addHeader("\"GenericService.h\"");
   instance->addHeader("\"ServiceDescriptor.h\"");
   addInstanceServiceHeaders(instance);
   addExtraServiceHeaders(instance);
   if (isSupportedMachineType(MachineType::GPU))
   {
      instance->addClass(getCompCategoryBaseName());
   }
   instance->addClass(getInstanceBaseName());

   CustomAttribute* data = new CustomAttribute("_data", getInstanceBaseName(), 
					       AccessType::PRIVATE);
   data->setPointer();
   attCup.reset(data);
   instance->addAttribute(attCup);

   CustomAttribute* serviceDescriptors = 
      new CustomAttribute(SERVICEDESCRIPTORS, 
			  "static std::vector<ServiceDescriptor>",
			  AccessType::PRIVATE);
   attCup.reset(serviceDescriptors);
   instance->addAttribute(attCup);

   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getPublisherName()));
   constructor->addParameter("Simulation& sim");
   constructor->addParameter(getInstanceBaseName() + "* data");
   constructor->setInitializationStr(
      "GeneratedPublisherBase(sim), _data(data)");
   if ((_instances.size() + _optionalInstanceServices.size() + 
	getExtraNumberOfServices()) > 0) {
      std::ostringstream constructorFB;
      constructorFB 
	 << TAB << "if (" << SERVICEDESCRIPTORS << ".size() == 0) {\n"
	 << getInstanceServiceDescriptors(TAB)
	 << getOptionalInstanceServiceDescriptors(TAB)
	 << getExtraServiceDescriptors(TAB)
	 << getExtraOptionalServiceDescriptors(TAB)
	 << TAB << "}\n";
      constructor->setFunctionBody(constructorFB.str());
   }

   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);
 
   // getServiceDescriptors method
   std::auto_ptr<Method> getServiceDescriptorsMethod(
      new Method("getServiceDescriptors", 
		 "const std::vector<ServiceDescriptor>&"));
   getServiceDescriptorsMethod->setVirtual();
   getServiceDescriptorsMethod->setConst();
   getServiceDescriptorsMethod->setFunctionBody(
      TAB + "return " + SERVICEDESCRIPTORS + ";\n");
   instance->addMethod(getServiceDescriptorsMethod);

   std::ostringstream extraSourceString;
   extraSourceString
      << "std::vector<ServiceDescriptor> " << getPublisherName() 
      << "::" << SERVICEDESCRIPTORS << ";\n";
   instance->addExtraSourceString(extraSourceString.str());

   // getName method
   std::auto_ptr<Method> getNameMethod(new Method("getName", "std::string"));
   getNameMethod->setVirtual();
   getNameMethod->setConst();
   getNameMethod->setFunctionBody(
      TAB + "return \"" + getName() + "Publisher\";\n");
   instance->addMethod(getNameMethod);

   // getDescription method
   std::auto_ptr<Method> getDescriptionMethod(new Method("getDescription", 
							 "std::string"));
   getDescriptionMethod->setVirtual();
   getDescriptionMethod->setConst();
   getDescriptionMethod->setFunctionBody(
      TAB + "return \"\";\n");
   instance->addMethod(getDescriptionMethod);

   // add duplicate method - standard methods are not added
   std::auto_ptr<Method> dupPublisher(new Method("duplicate", "void"));
   dupPublisher->addParameter("std::unique_ptr<Publisher>& dup");
   dupPublisher->setFunctionBody(
      TAB + "dup.reset(new " + getPublisherName() + "(*this));\n");
   dupPublisher->setVirtual();
   dupPublisher->setConst();
   instance->addMethod(dupPublisher);

   // createService method
   std::auto_ptr<Method> createServiceMethod(
      new Method("createService", "Service*"));
   createServiceMethod->setVirtual();
   createServiceMethod->setAccessType(AccessType::PROTECTED);
   createServiceMethod->addParameter(
      "const std::string& " + SERVICEREQUESTED);
   std::ostringstream createServiceMethodFB;
   createServiceMethodFB 
      << TAB << "Service* rval = 0;\n"
      << getInstanceServices(TAB)
      << getExtraServices(TAB)
      << TAB << "return rval;\n";
   createServiceMethod->setFunctionBody(
      createServiceMethodFB.str());
   instance->addMethod(createServiceMethod);

   // createOptionalService method
   std::auto_ptr<Method> createOptionalServiceMethod(
      new Method("createOptionalService", "Service*"));
   createOptionalServiceMethod->setVirtual();
   createOptionalServiceMethod->setAccessType(AccessType::PROTECTED);
   createOptionalServiceMethod->addParameter(
      "const std::string& " + SERVICEREQUESTED);
   std::ostringstream createOptionalServiceMethodFB;
   createOptionalServiceMethodFB 
      << TAB << "Service* rval = 0;\n"
      << getOptionalInstanceServices(TAB)
      << getExtraOptionalServices(TAB)
      << TAB << "return rval;\n";
   createOptionalServiceMethod->setFunctionBody(
      createOptionalServiceMethodFB.str());
   instance->addMethod(createOptionalServiceMethod);

   // getServiceNameWithInterface method
   std::auto_ptr<Method> getServiceNameWithInterfaceMethod(
      new Method("getServiceNameWithInterface", "std::string"));
   getServiceNameWithInterfaceMethod->setVirtual();
   getServiceNameWithInterfaceMethod->setAccessType(AccessType::PROTECTED);
   getServiceNameWithInterfaceMethod->addParameter(
      "const std::string& " + INTERFACENAME);
   getServiceNameWithInterfaceMethod->addParameter(
      "const std::string& " + SUBINTERFACENAME);
   getServiceNameWithInterfaceMethod->setFunctionBody(getServiceNameCode());
   instance->addMethod(getServiceNameWithInterfaceMethod);

   // Don't add the standard methods
   instance->addBasicDestructor();
   
   _classes.push_back(instance.release());  
}

void InterfaceImplementorBase::generateInstanceBase()
{
   std::auto_ptr<Class> instance(new Class(getInstanceBaseName()));
   //instance.addSupportMachineType(this->getSupportMachineType);

   instance->addHeader("\"" + getPublisherName() + ".h\"");
   instance->addHeader("\"" + getOutAttrPSetName() + ".h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<iostream>"); // for cerr in connections
   instance->addClass("Constant");
   instance->addClass("Variable");
   instance->addClass("VariableDescriptor");
   instance->addClass("Node");
   instance->addClass("Edge");
   instance->addClass("ConnectionIncrement");     // added by Jizhu Lu on 03/01/2006

   // Add the instances.
   //instance->set_gpu();
   if (isSupportedMachineType(MachineType::GPU))
   {
      std::auto_ptr<Method> setCC(
            new Method(SETCOMPCATEGORY_FUNC_NAME, "void"));
      //getPublisherMethod->setVirtual();
      std::string arg1 = "_" + REF_INDEX;
      setCC->addParameter("int " + arg1);
      setCC->addParameter(instance->getName() + COMPCATEGORY + "* cc_ptr");
      std::ostringstream setCCstream;
      setCCstream 
         << TAB << TAB << REF_INDEX << " = " << arg1 << "; " << REF_CC_OBJECT << " = cc_ptr;\n";
      setCC->setFunctionBody(setCCstream.str());
      setCC->setInline();
      MacroConditional gpuConditional(GPUCONDITIONAL);
      setCC->setMacroConditional(gpuConditional);
      instance->addMethod(setCC);
      std::auto_ptr<Method> getCC(
            new Method(GETCOMPCATEGORY_FUNC_NAME, instance->getName() + COMPCATEGORY + "*"));
      //getPublisherMethod->setVirtual();
      std::ostringstream getCCstream;
      getCCstream 
         << TAB << TAB << "return " << REF_CC_OBJECT << ";\n";
      getCC->setFunctionBody(getCCstream.str());
      getCC->setInline();
      getCC->setMacroConditional(gpuConditional);
      instance->addMethod(getCC);
      instance->addAttributes(getInstances(), AccessType::PROTECTED, false, true);
   }
   else{
      instance->addAttributes(getInstances(), AccessType::PROTECTED, false);
   }

   // Add the interface base classes, implement the get methods.
   setupInstanceInterfaces(instance);

   // Add the extra interface base classes, and implement their get 
   // methods, used for shared variables.
   setupExtraInterfaces(instance);

   // Add the friends
   FriendDeclaration publisherFriend(getPublisherName());
   instance->addFriendDeclaration(publisherFriend);

   // getServiceName method
   std::auto_ptr<Method> getServiceNameMethod(
      new Method("getServiceName", "const char*"));
   getServiceNameMethod->setVirtual();
   getServiceNameMethod->setConst();
   getServiceNameMethod->addParameter(PUBDATATYPE + " " + PUBDATANAME);
   std::ostringstream getServiceNameMethodFB;
   if (isSupportedMachineType(MachineType::GPU))
   {
      getServiceNameMethodFB
         << STR_GPU_CHECK_START
         << getInstanceServiceNames(TAB, MachineType::GPU)
         << getOptionalInstanceServiceNames(TAB)
         << getExtraServiceNames(TAB)
         << getExtraOptionalServiceNames(TAB)
         << "#else\n";
   }
   else{//leave blank
   }
   getServiceNameMethodFB
      << getInstanceServiceNames(TAB)
      << getOptionalInstanceServiceNames(TAB)
      << getExtraServiceNames(TAB)
      << getExtraOptionalServiceNames(TAB);
   if (isSupportedMachineType(MachineType::GPU))
   {
      getServiceNameMethodFB
         << STR_GPU_CHECK_END;
   }
   else{//leave blank
   }
   getServiceNameMethodFB
      << TAB << "return \"Error in Service Name!\";\n";
   getServiceNameMethod->setFunctionBody(
      getServiceNameMethodFB.str());
   instance->addMethod(getServiceNameMethod);
   
   // getServiceDescription method
   std::auto_ptr<Method> getServiceDescriptionMethod(
      new Method("getServiceDescription", "const char*"));
   getServiceDescriptionMethod->setVirtual();
   getServiceDescriptionMethod->setConst();
   getServiceDescriptionMethod->addParameter(PUBDATATYPE + " " + PUBDATANAME);
   std::ostringstream getServiceDescriptionMethodFB;
   if (isSupportedMachineType(MachineType::GPU))
   {
      getServiceDescriptionMethodFB
         << STR_GPU_CHECK_START
         << getInstanceServiceDescriptions(TAB, MachineType::GPU)
         << getOptionalInstanceServiceDescriptions(TAB)
         << getExtraServiceDescriptions(TAB)
         << getExtraOptionalServiceDescriptions(TAB)
         <<  "#else\n";
   }
   getServiceDescriptionMethodFB
      << getInstanceServiceDescriptions(TAB)
      << getOptionalInstanceServiceDescriptions(TAB)
      << getExtraServiceDescriptions(TAB)
      << getExtraOptionalServiceDescriptions(TAB);
   if (isSupportedMachineType(MachineType::GPU))
   {
   getServiceDescriptionMethodFB
      << STR_GPU_CHECK_END;
   }
   getServiceDescriptionMethodFB
      << TAB << "return \"Error in Service Description!\";\n";
   getServiceDescriptionMethod->setFunctionBody(
      getServiceDescriptionMethodFB.str());
   instance->addMethod(getServiceDescriptionMethod);
   
   addGetPublisherMethod(*(instance.get()));

   mdl::addOptionalServicesToClass(*(instance.get()), 
				   _optionalInstanceServices);

   if (getType()!="Constant" && getType()!="Edge")
      addDistributionCodeToIB(*(instance.get()));

   addExtraInstanceBaseMethods(*(instance.get()));
   instance->addStandardMethods();
   _classes.push_back(instance.release());  
}

void InterfaceImplementorBase::generateInstanceProxy()
{
   if (getType()=="Constant" || getType()=="Edge") return;
   std::auto_ptr<Class> instance(new Class(getInstanceProxyName()));

   MacroConditional mpiConditional(MPICONDITIONAL);

   instance->setMacroConditional(mpiConditional);

   //instance->addHeader("\"" + getPublisherName() + ".h\"");
   // for now
   instance->addClass("Publisher");
   instance->addHeader("\"" + getOutAttrPSetName() + ".h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<iostream>"); // for cerr in connections
   instance->addHeader("<cassert>"); // for unimplemented functions
   instance->addHeader("\"" + getInstanceProxyDemarshallerName()+ ".h\"");
   instance->addHeader("\"DemarshallerInstance.h\"");
   instance->addClass("Constant");
   instance->addClass("Variable");
   instance->addClass("Node");
   instance->addClass("Edge");

   // Add the instances.
   instance->addAttributes(getInstances(), AccessType::PROTECTED, true);

   setInterfaceImplementors();

   // Add the interface base classes, implement the get methods.
   setupProxyInterfaces(instance);

   // Add the extra interface base classes, and implement their get 
   // methods, used for shared variables.
   setupExtraInterfaces(instance);

   // Add the friends
   FriendDeclaration publisherFriend(getPublisherName());
   instance->addFriendDeclaration(publisherFriend);

   if (_instancePhases) {     // added by Jizhu Lu on 06/30/2006 to fix a bug of "segmentation fault"
      std::vector<Phase*>::iterator piter, pbegin =  _instancePhases->begin();
      std::vector<Phase*>::iterator pend =  _instancePhases->end();
      for (piter = pbegin; piter != pend; ++piter) {
         if ((*piter)->hasPackedVariables()) {
            std::auto_ptr<Method> demarshaller(
   		    new Method(PREFIX+"recv_"+(*piter)->getName()+"_demarshaller", "void"));
            demarshaller->setAccessType(AccessType::PUBLIC);
            demarshaller->setMacroConditional(mpiConditional);
            demarshaller->addParameter("std::unique_ptr<" + getInstanceProxyDemarshallerName() + "> &ap");
            demarshaller->setStatic();
            std::ostringstream funBody;
            funBody << TAB << "PhaseDemarshaller_" << (*piter)->getName() << "* di = new PhaseDemarshaller_" 
   		 << (*piter)->getName() << "();\n"
   		 << TAB << "ap.reset(di);\n"; 
            demarshaller->setFunctionBody(funBody.str());
            instance->addMethod(demarshaller);
         }
      }
   }

   std::auto_ptr<Method> initializeProxyDemarshaller(
      new Method(PREFIX + "recv_FLUSH_LENS_demarshaller", "void"));
   initializeProxyDemarshaller->setAccessType(AccessType::PUBLIC);
   initializeProxyDemarshaller->setMacroConditional(mpiConditional); 
   initializeProxyDemarshaller->addParameter("std::unique_ptr<" + getInstanceProxyDemarshallerName() + "> &ap");
   initializeProxyDemarshaller->setStatic();
   std::ostringstream funBody;
   funBody << TAB << "PhaseDemarshaller_FLUSH_LENS *di = new PhaseDemarshaller_FLUSH_LENS();\n"
	   << TAB << "ap.reset(di);\n"; 

   initializeProxyDemarshaller->setFunctionBody(funBody.str());
   instance->addMethod(initializeProxyDemarshaller);
  
   addExtraInstanceProxyMethods(*(instance.get()));
   instance->addStandardMethods();

   // Demarshaller classes

   //**** Proxy specific DemarshallerInstance base
   std::auto_ptr<Class> demarshallerInstanceBase(new Class(getInstanceProxyDemarshallerName())); 
   demarshallerInstanceBase->setAlternateFileName(getInstanceProxyName() + "Demarshaller");
   demarshallerInstanceBase->setMacroConditional(mpiConditional);
   demarshallerInstanceBase->addClass(getInstanceProxyName());
   demarshallerInstanceBase->addHeader("\"" + getInstanceProxyDemarshallerName() + ".h\"");
   demarshallerInstanceBase->addHeader("\"StructDemarshallerBase.h\"");
   std::auto_ptr<BaseClass> demarshallerBaseBase(new BaseClass("StructDemarshallerBase"));
   demarshallerInstanceBase->addBaseClass(demarshallerBaseBase);

   // Constructors
   std::auto_ptr<ConstructorMethod> baseConstructor1(new ConstructorMethod(getInstanceProxyDemarshallerName()));
   std::auto_ptr<ConstructorMethod> baseConstructor2(new ConstructorMethod(getInstanceProxyDemarshallerName()));
   baseConstructor1->setInitializationStr("_proxy(0)");
   baseConstructor2->addParameter(getInstanceProxyName() + "* p");
   baseConstructor2->setInitializationStr("_proxy(p)");
   std::auto_ptr<Method> baseConsToIns1(baseConstructor1.release());
   std::auto_ptr<Method> baseConsToIns2(baseConstructor2.release());
   baseConsToIns1->setInline();
   baseConsToIns2->setInline();
   demarshallerInstanceBase->addMethod(baseConsToIns1);
   demarshallerInstanceBase->addMethod(baseConsToIns2);

   // setDestinationBase Method
   std::auto_ptr<Method> setDestinationBaseMethod(new Method("setDestination", "void"));
   setDestinationBaseMethod->setInline();
   setDestinationBaseMethod->setPureVirtual();
   setDestinationBaseMethod->addParameter(getInstanceProxyName()+" *proxy");
   demarshallerInstanceBase->addMethod(setDestinationBaseMethod);

   // proxyDestination attribute
   CustomAttribute* proxyDestination = new CustomAttribute("_proxy", getInstanceProxyName(), AccessType::PROTECTED);
   proxyDestination->setPointer();   
   std::auto_ptr<Attribute> proxyDestinationAp(proxyDestination);
   demarshallerInstanceBase->addAttribute(proxyDestinationAp);

   demarshallerInstanceBase->addBasicInlineDestructor();

   //**** member  initialize demarshaller classes
   std::auto_ptr<Class> demarshallerInstance(new Class("PhaseDemarshaller_FLUSH_LENS")); 
   std::string baseName = getInstanceProxyDemarshallerName();
   std::auto_ptr<BaseClass> demarshallerBase(new BaseClass(baseName));
   demarshallerInstance->addBaseClass(demarshallerBase);

   // Constructors
   std::auto_ptr<ConstructorMethod> constructor1(new ConstructorMethod("PhaseDemarshaller_FLUSH_LENS"));
   std::auto_ptr<ConstructorMethod> constructor2(new ConstructorMethod("PhaseDemarshaller_FLUSH_LENS"));

   std::list<const DataType*> allVars;
   std::vector<Phase*>::iterator phaseIter, phaseEnd;
   if (_instancePhases) {
      phaseEnd = _instancePhases->end();
      for (phaseIter = _instancePhases->begin(); phaseIter != phaseEnd; ++phaseIter) {
        if ((*phaseIter)->hasPackedVariables()) {
          std::vector<const DataType*> vars = (*phaseIter)->getPackedVariables();
          std::vector<const DataType*>::iterator varsIter, varsEnd = vars.end();
          for (varsIter = vars.begin(); varsIter != varsEnd; ++varsIter) {
             allVars.push_back(*varsIter);
          }
        }
      }
   }
   allVars.sort();
   allVars.unique();
   std::ostringstream constructorFB;
   std::ostringstream initString1;
   std::ostringstream initString2;
   std::ostringstream setDestinationMethodFB;
   initString1 << (baseName + "()");
   initString2 << (baseName + "(proxy)");
   setDestinationMethodFB << TAB << TAB << TAB << "_proxy = proxy;\n";

   std::list<std::string> varNameList;
   std::list<const DataType*>::iterator varsIter, varsEnd = allVars.end();
   varsIter = allVars.begin();
   while (varsIter != varsEnd) {
     varNameList.push_back((*varsIter)->getName());
     ++varsIter;
   }
   varNameList.sort();

   bool parsingError = false;
   std::vector<DataType*>::iterator it, end = _interfaceImplementors.end();
   for (it = _interfaceImplementors.begin(); it != end; ++it) {
      initString1<<", ";
      initString2<<", ";
      std::string varName = (*it)->getName();
      std::string varDesc = (*it)->getDescriptor();
      if (!binary_search(varNameList.begin(), varNameList.end(), varName)) {
         std::cerr << "Warning: variable \'" << (*it)->getName() << "\' used in interface is missing in phase's changing variable list!" << std::endl;
         parsingError = true;
      }
      constructorFB << TAB << TAB << TAB << "_demarshallers.push_back(&" << varName << "Demarshaller);\n";
      initString1 << varName << "Demarshaller()";
      initString2 << varName << "Demarshaller(&(proxy->" << varName <<"))";
      setDestinationMethodFB << TAB << TAB << TAB << varName << "Demarshaller.setDestination(&(_proxy->" << varName << "));\n";
    
      CustomAttribute* demarshaller;
      if ((*it)->isTemplateDemarshalled()) demarshaller = new CustomAttribute(varName + "Demarshaller", 
							    "DemarshallerInstance< " + varDesc + " >", AccessType::PRIVATE);
      else {
	demarshaller = new CustomAttribute(varName + "Demarshaller",
					   "CG_" + varDesc + "Demarshaller", AccessType::PRIVATE);
	std::set<std::string> subStructTypes;
	(*it)->getSubStructDescriptors(subStructTypes);
	std::set<std::string>::iterator stypeIter, stypeEnd = subStructTypes.end();
	for (stypeIter=subStructTypes.begin(); stypeIter!=stypeEnd; ++stypeIter) {
	  instance->addHeader("\"CG_" + (*stypeIter) + "Demarshaller.h\"");
	}	
      }
      std::auto_ptr<Attribute> demarshallerAp(demarshaller);
      demarshallerInstance->addAttribute(demarshallerAp);
   }   
   setDestinationMethodFB << TAB << TAB << TAB << "reset();\n";
   if (parsingError) {
      std::cerr << "\nWarning: Some variables used in interface are missing in phase's changing variable list!\n" << std::endl;
   }

   constructor1->setInitializationStr(initString1.str());
   constructor1->setFunctionBody(constructorFB.str());
   constructor1->setInline();
   constructor2->setInitializationStr(initString2.str());
   constructor2->setFunctionBody(constructorFB.str());
   constructor2->addParameter(getInstanceProxyName() + "* proxy");
   constructor2->setInline();

   std::auto_ptr<Method> consToIns1(constructor1.release());
   std::auto_ptr<Method> consToIns2(constructor2.release());
   demarshallerInstance->addMethod(consToIns1);
   demarshallerInstance->addMethod(consToIns2);
   
   std::auto_ptr<Method> setDestinationMethod(new Method("setDestination", "void"));
   setDestinationMethod->setInline();
   setDestinationMethod->addParameter(getInstanceProxyName()+" *proxy");
   setDestinationMethod->setFunctionBody(setDestinationMethodFB.str());
   demarshallerInstance->addMethod(setDestinationMethod);

   demarshallerInstance->addBasicInlineDestructor();

   instance->addMemberClass(demarshallerInstance, AccessType::PRIVATE);

   //**** Phase specific member demarshaller classes
   if (_instancePhases) {   // added by Jizhu Lu on 06/30/2006 to fix a bug of "segmentation fault"
      for (phaseIter = _instancePhases->begin(); phaseIter != phaseEnd; ++phaseIter) {
        if ((*phaseIter)->hasPackedVariables()) {
          std::string phaseName = (*phaseIter)->getName();
          std::auto_ptr<Class> demarshallerInstance(new Class("PhaseDemarshaller_" + phaseName)); 
          std::string baseName = getInstanceProxyDemarshallerName();
          std::auto_ptr<BaseClass> demarshallerBase(new BaseClass(baseName));
          demarshallerInstance->addBaseClass(demarshallerBase);
   
          // Constructors
          std::auto_ptr<ConstructorMethod> constructor1(new ConstructorMethod("PhaseDemarshaller_" + phaseName));
          std::auto_ptr<ConstructorMethod> constructor2(new ConstructorMethod("PhaseDemarshaller_" + phaseName));
   
          std::vector<const DataType*> vars = (*phaseIter)->getPackedVariables();
   
          std::ostringstream constructorFB;
          std::ostringstream initString1;
          std::ostringstream initString2;
          std::ostringstream setDestinationMethodFB;
          initString1 << (baseName + "()");
          initString2 << (baseName + "(proxy)");
          setDestinationMethodFB << TAB << TAB << TAB << "_proxy = proxy;\n";
          if (allVars.size() != 0) {
   	     initString1<<", ";
   	     initString2<<", ";
          }
   
          std::vector<const DataType*>::iterator varsIter, varsEnd = vars.end();
          varsIter = vars.begin();
          while (varsIter != varsEnd) {
   	     std::string varName = (*varsIter)->getName();
   	     std::string varDesc = (*varsIter)->getDescriptor();
   	     constructorFB << TAB << TAB << TAB << "_demarshallers.push_back(&" << varName << "Demarshaller);\n";
   	     initString1 << varName << "Demarshaller()";
   	     initString2 << varName << "Demarshaller(&(proxy->" << varName <<"))";
   	     setDestinationMethodFB << TAB << TAB << TAB << varName << "Demarshaller.setDestination(&(_proxy->" << varName << "));\n";
   
	     CustomAttribute* demarshaller;
	     if ((*varsIter)->isTemplateDemarshalled()) demarshaller = new CustomAttribute(varName + "Demarshaller", 
								      "DemarshallerInstance< " + varDesc + " >", AccessType::PRIVATE);
	     else demarshaller = new CustomAttribute(varName + "Demarshaller",
						     "CG_" + varDesc + "Demarshaller", AccessType::PRIVATE);
	     
   	     std::auto_ptr<Attribute> demarshallerAp(demarshaller);
   	     demarshallerInstance->addAttribute(demarshallerAp);
   
   	     if (++varsIter != varsEnd) {
   	        initString1 << ", ";
   	        initString2 << ", ";
   	     }
          }
          setDestinationMethodFB << TAB << TAB << TAB << "reset();\n";
   
          constructor1->setInitializationStr(initString1.str());
          constructor1->setFunctionBody(constructorFB.str());
          constructor1->setInline();
          constructor2->setInitializationStr(initString2.str());
          constructor2->setFunctionBody(constructorFB.str());
          constructor2->addParameter(getInstanceProxyName() + "* proxy");
          constructor2->setInline();
   
          std::auto_ptr<Method> consToIns1(constructor1.release());
          std::auto_ptr<Method> consToIns2(constructor2.release());
          demarshallerInstance->addMethod(consToIns1);
          demarshallerInstance->addMethod(consToIns2);
      
          std::auto_ptr<Method> setDestinationMethod(new Method("setDestination", "void"));
          setDestinationMethod->setInline();
          setDestinationMethod->addParameter(getInstanceProxyName()+" *proxy");
          setDestinationMethod->setFunctionBody(setDestinationMethodFB.str());
          demarshallerInstance->addMethod(setDestinationMethod);
   
          demarshallerInstance->addBasicInlineDestructor();
          instance->addMemberClass(demarshallerInstance, AccessType::PRIVATE);
        }
      }
   }
   // End Demarshaller classes

   // add the proxy class to the base
   _classes.push_back(demarshallerInstanceBase.release());  
   _classes.push_back(instance.release());  
}

void InterfaceImplementorBase::addInstanceServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
   instance->addDataTypeHeaders(_instances);
}

void InterfaceImplementorBase::addOptionalInstanceServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
   instance->addDataTypeHeaders(_optionalInstanceServices);
}

// will be implemented in derived classes.
void InterfaceImplementorBase::addExtraServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
}

// will be implemented in derived classes.
void InterfaceImplementorBase::addExtraOptionalServiceHeaders(
   std::auto_ptr<Class>& instance) const
{
}

std::string InterfaceImplementorBase::getInstanceServices(
   const std::string& tab) const
{
	//TUAN: IDEA - currently the body is generated as a sequence of 'if-statement'
	//        which requires so many 'conditional-test' while only once is 'true'
	// E.g.:CG_HodgkinHuxleyVoltageJunctionPublisher.C
	/*
   Service* rval = 0;
   if (serviceRequested == "dimensions") {
      rval = new GenericService< ShallowArray< DimensionStruct* > >(_data, &(_data->dimensions));
      _services.push_back(rval);
      return rval;
   }
   if (serviceRequested == "area") {
      rval = new GenericService< float >(_data, &(_data->area));
      _services.push_back(rval);
      return rval;
   }
   if (serviceRequested == "gAxial") {
      rval = new GenericService< ShallowArray< float > >(_data, &(_data->gAxial));
      _services.push_back(rval);
      return rval;
   }
   if (serviceRequested == "Vcur") {
      rval = new GenericService< float >(_data, &(_data->Vcur));
      _services.push_back(rval);
      return rval;
   }
   if (serviceRequested == "Vnew") {
      rval = new GenericService< ShallowArray< float > >(_data, &(_data->Vnew));
      _services.push_back(rval);
      return rval;
   }
	*/
	// Can we automate the code generation using
	// 'dictionary' or 'switch statement' here?
   return createServices(_instances, tab);
}

std::string InterfaceImplementorBase::getOptionalInstanceServices(
   const std::string& tab) const
{
   return createOptionalServices(_optionalInstanceServices, tab);
}

// Will be implemented in the derived classes.
std::string InterfaceImplementorBase::getExtraServices(
   const std::string& tab) const
{
   return "";
}

// Will be implemented in the derived classes.
std::string InterfaceImplementorBase::getExtraOptionalServices(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getInstanceServiceNames(
   const std::string& tab, MachineType mach_type) const
{
   return createServiceNames(_instances, tab, mach_type);   
}

std::string InterfaceImplementorBase::getOptionalInstanceServiceNames(
   const std::string& tab) const
{
   return createOptionalServiceNames(_optionalInstanceServices, tab);   
}

std::string InterfaceImplementorBase::getExtraServiceNames(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getExtraOptionalServiceNames(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getInstanceServiceDescriptions(
   const std::string& tab,
   MachineType mach_type
   ) const
{
   return createServiceDescriptions(_instances, tab, mach_type);   
}

std::string InterfaceImplementorBase::getOptionalInstanceServiceDescriptions(
   const std::string& tab) const
{
   return createOptionalServiceDescriptions(_optionalInstanceServices, tab);   
}

std::string InterfaceImplementorBase::getExtraServiceDescriptions(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getExtraOptionalServiceDescriptions(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getInstanceServiceDescriptors(
   const std::string& tab) const
{
   return createServiceDescriptors(_instances, tab);   
}

std::string InterfaceImplementorBase::getOptionalInstanceServiceDescriptors(
   const std::string& tab) const
{
   return createOptionalServiceDescriptors(_optionalInstanceServices, tab);   
}

std::string InterfaceImplementorBase::getExtraServiceDescriptors(
   const std::string& tab) const
{
   return "";
}

std::string InterfaceImplementorBase::getExtraOptionalServiceDescriptors(
   const std::string& tab) const
{
   return "";
}


std::string InterfaceImplementorBase::createServices(
   const MemberContainer<DataType>& members, const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      if (isSupportedMachineType(MachineType::GPU))
      {
        os << STR_GPU_CHECK_START;
        os << (*it).second->getServiceString(tab, MachineType::GPU)
            << "#else\n";
      }
      os << (*it).second->getServiceString(tab);
      if (isSupportedMachineType(MachineType::GPU))
      {
         os << STR_GPU_CHECK_END;
      }
   }
   return os.str();
}

std::string InterfaceImplementorBase::createOptionalServices(
   const MemberContainer<DataType>& members, const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getOptionalServiceString(tab);
   }
   return os.str();
}

std::string InterfaceImplementorBase::createServiceNames(
   const MemberContainer<DataType>& members, 
   const std::string& tab,
   MachineType mach_type) const
{
   std::ostringstream os;
   if (mach_type == MachineType::CPU)
   {
      MemberContainer<DataType>::const_iterator it, end = members.end();
      for (it = members.begin(); it != end; ++it) {
         os << (*it).second->getServiceNameString(tab);
      }
   }
   else if (mach_type == MachineType::GPU)
   {
      MemberContainer<DataType>::const_iterator it, end = members.end();
      for (it = members.begin(); it != end; ++it) {
         os << (*it).second->getServiceNameString(tab, mach_type);
      }
   }
   return os.str();
}

std::string InterfaceImplementorBase::createOptionalServiceNames(
   const MemberContainer<DataType>& members, 
   const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getOptionalServiceNameString(tab);
   }
   return os.str();
}

std::string InterfaceImplementorBase::createServiceDescriptions(
   const MemberContainer<DataType>& members, 
   const std::string& tab,
   MachineType mach_type
   ) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getServiceDescriptionString(tab, mach_type);
   }
   return os.str();
}

std::string InterfaceImplementorBase::createServiceDescriptors(
   const MemberContainer<DataType>& members, 
   const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getServiceDescriptorString(tab);
   }
   return os.str();
}

std::string InterfaceImplementorBase::createOptionalServiceDescriptors(
   const MemberContainer<DataType>& members, 
   const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getOptionalServiceDescriptorString(tab);
   }
   return os.str();
}

std::string InterfaceImplementorBase::createOptionalServiceDescriptions(
   const MemberContainer<DataType>& members, 
   const std::string& tab) const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; ++it) {
      os << (*it).second->getOptionalServiceDescriptionString(tab);
   }
   return os.str();
}


void InterfaceImplementorBase::setupInstanceInterfaces(
   std::auto_ptr<Class>& instance)
{
   std::auto_ptr<BaseClass> baseClass;
   MemberContainer<MemberToInterface>::iterator it, 
      end = _interfaces.end();
   std::string baseName;
   for (it = _interfaces.begin(); it != end; ++it) {
      baseName = it->second->getInterface()->getName();
      baseClass.reset(new BaseClass(baseName));
      instance->addBaseClass(baseClass);
      instance->addHeader("\"" + baseName + ".h\"");
      // add the accessor methods for interfaces
      it->second->setupAccessorMethods(*instance.get());
   }
}

void InterfaceImplementorBase::setupProxyInterfaces(
   std::auto_ptr<Class>& instance)
{
   std::auto_ptr<BaseClass> baseClass;
   MemberContainer<MemberToInterface>::iterator it, 
      end = _interfaces.end();
   std::string baseName;
   for (it = _interfaces.begin(); it != end; ++it) {
      baseName = it->second->getInterface()->getName();
      baseClass.reset(new BaseClass(baseName));
      instance->addBaseClass(baseClass);
      instance->addHeader("\"" + baseName + ".h\"");
      // add the accessor methods for interfaces
      it->second->setupProxyAccessorMethods(*instance.get());
   }
}

std::string InterfaceImplementorBase::getAddConnectionFunctionBody(
   Connection::ComponentType componentType, 
   Connection::DirectionType directionType) const
{
   std::ostringstream os;
   std::string connectionAdd;
   std::string componentName;
   std::string psetType;
   std::string psetName;
   if (directionType == Connection::_PRE) {
      connectionAdd = "Pre";
      psetName = INATTRPSETNAME;
      psetType = getInAttrPSetName();
   } else { // Connection::_POST
      connectionAdd = "Post";
      psetName = OUTATTRPSETNAME;
      psetType = getOutAttrPSetName();
   }
   connectionAdd += Connection::getStringForComponentType(componentType);
   componentName = Connection::getParameterNameForComponentType(componentType);
   os << getAddConnectionFunctionBodyExtra(componentType, directionType,
					   componentName, psetType, psetName);
   os << TAB << "checkAndAdd" << connectionAdd << "(" 
      << componentName << ");\n";
	 if (connectionAdd == "PreNode" )
   	 os << TAB << "assert(noPredicateMatch || matchPredicateAndCast);\n";
   return os.str();
}

std::string InterfaceImplementorBase::getAddPostEdgeFunctionBody() const
{
   return getAddConnectionFunctionBody(Connection::_EDGE, 
				       Connection::_POST);
}

std::string InterfaceImplementorBase::getAddPostNodeFunctionBody() const
{
   return getAddConnectionFunctionBody(Connection::_NODE, 
				       Connection::_POST);
}

std::string InterfaceImplementorBase::getAddPostVariableFunctionBody() const
{
   return getAddConnectionFunctionBody(Connection::_VARIABLE, 
				       Connection::_POST);
}

bool InterfaceImplementorBase::isMemberToInterface(
   const DataType& member) const
{
   MemberContainer<MemberToInterface>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; ++it) {
      if (it->second->hasMemberDataType(member.getName())) {
	 return true;
      }
   }
   return false;
}

void InterfaceImplementorBase::setInterfaceImplementors()
{
   if (_interfaceImplementors.size() > 0) { // already set
      return;
   }
   MemberContainer<DataType>::iterator it, end = _instances.end();
   for (it= _instances.begin(); it != end; ++it) {
      if (isMemberToInterface(*it->second)) {
	 _interfaceImplementors.push_back(it->second);
      }
   }
}

std::string InterfaceImplementorBase::getServiceNameCode() const
{
   std::ostringstream os;

   MemberContainer<MemberToInterface>::const_iterator it, 
      end = _interfaces.end();
   for(it = _interfaces.begin(); it != end; ++it) {
      os << it->second->getServiceNameCode(TAB);      
   }
   os << TAB << "return \"\";\n";
   return os.str();
}

void InterfaceImplementorBase::addMappingToInterface(
   const std::string& interfaceName, const std::string& interfaceMemberName, 
   std::auto_ptr<DataType>& dtToInsert, bool ampersand) 
{
   MemberToInterface* curMti;
   try {
      curMti = _interfaces.getMember(interfaceName);
   } catch (NotFoundException& e) {
      std::cerr << "In " << getName() << " interface " << e.getError() 
		<< std::endl;
      e.setError("");
      throw;
   }
   try {
      curMti->addMapping(interfaceMemberName, dtToInsert, ampersand);
   } catch(GeneralException& e) {
      std::cerr << "In " << getName() << e.getError() << std::endl;
      e.setError("");
      throw;
   }
}

void InterfaceImplementorBase::checkInstanceVariableNameSpace(
   const std::string& name) const 
{
   if (_instances.exists(name)) {
      std::ostringstream os;
      os << name + " is already included as an instance variable.";
      throw SyntaxErrorException(os.str()); 
   } else if (_optionalInstanceServices.exists(name)) {
      std::ostringstream os;
      os << name + " is already included as an optional service.";
      throw SyntaxErrorException(os.str()); 
   }
}

void InterfaceImplementorBase::addGetPublisherMethod(Class& instance) const
{
   // getPublisher method
   std::auto_ptr<Method> getPublisherMethod(
      new Method("getPublisher", "Publisher*"));
   getPublisherMethod->setVirtual();
   std::ostringstream getPublisherFB;
   getPublisherFB 
      << TAB << "if (_publisher == 0) {\n"
      << TAB << TAB << "_publisher = new " << getInstanceBaseName()
      << "Publisher(getSimulation(), this);\n"
      << TAB << "}\n"
      << TAB << "return _publisher;\n";
   getPublisherMethod->setFunctionBody(getPublisherFB.str());
   instance.addMethod(getPublisherMethod);
}

void InterfaceImplementorBase::addDistributionCodeToIB(Class& instance)
{
   setInterfaceImplementors();

   instance.addHeader("\"Marshall.h\"", MPICONDITIONAL);
   instance.addHeader("\"" + OUTPUTSTREAM + ".h\"", MPICONDITIONAL);
   instance.addHeader("\"" + getInstanceProxyName() + "Demarshaller.h\"", MPICONDITIONAL);

   MacroConditional mpiConditional(MPICONDITIONAL);
   if (_instancePhases) {
      std::vector<Phase*>::iterator piter, pbegin =  _instancePhases->begin();
      std::vector<Phase*>::iterator pend =  _instancePhases->end();
      for (piter = pbegin; piter != pend; ++piter) {
         if ((*piter)->hasPackedVariables()) {
            std::auto_ptr<Method> sender(
               new Method(PREFIX+"send_"+(*piter)->getName(), "void"));
            sender->setAccessType(AccessType::PROTECTED);
            sender->setInline();
	    sender->setMacroConditional(mpiConditional);
            sender->addParameter(OUTPUTSTREAM + "* stream");
            sender->setConst();
            std::ostringstream funBody;
            std::vector<const DataType*>::iterator pviter, pvbegin = ((*piter)->getPackedVariables()).begin();
            std::vector<const DataType*>::iterator pvend = ((*piter)->getPackedVariables()).end();
            std::map<std::string,int> typeMarshaller;
            std::map<std::string,int>::iterator typeMarshallerIter;
            int miSN, typeSN = 0;
            for (pviter = pvbegin; pviter != pvend; ++pviter) {
               if ((typeMarshallerIter=typeMarshaller.find((*pviter)->getDescriptor())) == typeMarshaller.end()) {
                  miSN = typeSN++;
                  typeMarshaller[(*pviter)->getDescriptor()] = miSN;
		  funBody << TAB << TAB << TAB;
		  if ((*pviter)->isTemplateMarshalled())
		    funBody << "MarshallerInstance<" << (*pviter)->getDescriptor() << " > mi" << miSN << ";\n";
		  else
		    funBody << "CG_" << (*pviter)->getDescriptor() << "MarshallerInstance mi" << miSN <<";\n";
               } else
                  miSN = (*typeMarshallerIter).second;
               funBody << TAB << TAB << TAB << "mi" << miSN << ".marshall(stream, ";
	       if ((*pviter)->isPointer()) funBody << "*";
	       funBody << (*pviter)->getName() << ");\n";
            }   
            sender->setFunctionBody(funBody.str());
            instance.addMethod(sender);

            std::auto_ptr<Method> getSendType(
               new Method(PREFIX+"getSendType_"+(*piter)->getName(), "void"));
            getSendType->setAccessType(AccessType::PROTECTED);
            getSendType->setMacroConditional(mpiConditional);
            getSendType->addParameter("std::vector<int>& blengths");
            getSendType->addParameter("std::vector<MPI_Aint>& blocs");
            getSendType->setConst();
            funBody.str("");
	    funBody.clear();
            typeMarshaller.clear();
            typeSN = 0;
            for (pviter = pvbegin; pviter != pvend; ++pviter) {
               if ((typeMarshallerIter=typeMarshaller.find((*pviter)->getDescriptor())) == typeMarshaller.end()) {
		 miSN = typeSN++;
		 typeMarshaller[(*pviter)->getDescriptor()] = miSN;
		 if ((*pviter)->isTemplateMarshalled())
		   funBody << TAB << "MarshallerInstance<" << (*pviter)->getDescriptor() << " > mi" << miSN << ";\n";
		 else
		   funBody << TAB << "CG_" << (*pviter)->getDescriptor() << "MarshallerInstance mi" << miSN <<";\n";
               } else
                  miSN = (*typeMarshallerIter).second;

               if (isSupportedMachineType(MachineType::GPU))
               {
                  funBody 
                     << STR_GPU_CHECK_START;
                  funBody << TAB << "mi" << miSN << ".getBlocks(blengths, blocs, ";
                  if ((*pviter)->isPointer()) funBody << "*";
                  funBody 
                     << (*pviter)->getName(MachineType::GPU) << ");\n"
                     << "#else\n";
               }
               funBody << TAB << "mi" << miSN << ".getBlocks(blengths, blocs, ";
	       if ((*pviter)->isPointer()) funBody << "*";
	       funBody << (*pviter)->getName() << ");\n";
               if (isSupportedMachineType(MachineType::GPU))
               {
                  funBody 
                     << STR_GPU_CHECK_END;
               }
            }   
            getSendType->setFunctionBody(funBody.str());
            instance.addMethod(getSendType);
         }
      }
   }

   std::auto_ptr<Method> initializeProxySender(
      new Method(PREFIX + "send_FLUSH_LENS", "void"));
   initializeProxySender->setAccessType(AccessType::PROTECTED);
   initializeProxySender->setMacroConditional(mpiConditional);
   initializeProxySender->addParameter(OUTPUTSTREAM + "* stream");
   initializeProxySender->setConst();
   std::ostringstream funBody;
   std::map<std::string,int> typeMarshaller;
   std::map<std::string,int>::iterator typeMarshallerIter;
   int miSN, typeSN = 0;
   std::vector<DataType*>::iterator it, end = _interfaceImplementors.end();
   for (it = _interfaceImplementors.begin(); it != end; ++it) {
     std::set<std::string> subStructTypes;
     (*it)->getSubStructDescriptors(subStructTypes);
     std::set<std::string>::iterator stypeIter, stypeEnd = subStructTypes.end();
     for (stypeIter=subStructTypes.begin(); stypeIter!=stypeEnd; ++stypeIter)
       instance.addHeader("\"CG_" + (*stypeIter) + "MarshallerInstance.h\"",MPICONDITIONAL);
     if ((typeMarshallerIter=typeMarshaller.find((*it)->getDescriptor())) == typeMarshaller.end()) {
       miSN = typeSN++;
       typeMarshaller[(*it)->getDescriptor()] = miSN;
       funBody << TAB;
       if ((*it)->isTemplateMarshalled())
	 funBody << "MarshallerInstance<" << (*it)->getDescriptor() << " > mi" << miSN << ";\n";
       else
	 funBody << "CG_" << (*it)->getDescriptor() << "MarshallerInstance mi" << miSN <<";\n";
     } else
       miSN = (*typeMarshallerIter).second;

      if (isSupportedMachineType(MachineType::GPU))
      {
         funBody 
            << STR_GPU_CHECK_START;
         funBody << TAB << "mi" << miSN << ".marshall(stream, ";
         if ((*it)->isPointer()) funBody << "*";
         funBody << (*it)->getName() << ");\n"
            << "#else\n";
      }
     funBody << TAB << "mi" << miSN << ".marshall(stream, ";
     if ((*it)->isPointer()) funBody << "*";
     funBody << (*it)->getName() << ");\n";
      if (isSupportedMachineType(MachineType::GPU))
      {
         funBody 
            << STR_GPU_CHECK_END;
      }
   }
   initializeProxySender->setFunctionBody(funBody.str());
   instance.addMethod(initializeProxySender);

   std::auto_ptr<Method> initializeProxyGetSendType(
      new Method(PREFIX + "getSendType_FLUSH_LENS", "void"));
   initializeProxyGetSendType->setAccessType(AccessType::PROTECTED);
   initializeProxyGetSendType->setMacroConditional(mpiConditional);
   initializeProxyGetSendType->addParameter("std::vector<int>& blengths");
   initializeProxyGetSendType->addParameter("std::vector<MPI_Aint>& blocs");
   initializeProxyGetSendType->setConst();
   funBody.str("");
   funBody.clear();
   typeMarshaller.clear();
   typeSN = 0;
   for (it = _interfaceImplementors.begin(); it != end; ++it) {
      if ((typeMarshallerIter=typeMarshaller.find((*it)->getDescriptor())) == typeMarshaller.end()) {
         miSN = typeSN++;
         typeMarshaller[(*it)->getDescriptor()] = miSN;
	 funBody << TAB;
	 if ((*it)->isTemplateMarshalled())
	   funBody << "MarshallerInstance<" << (*it)->getDescriptor() << " > mi" << miSN << ";\n";
	 else
	   funBody << "CG_" << (*it)->getDescriptor() << "MarshallerInstance mi" << miSN <<";\n";
      } else
         miSN = (*typeMarshallerIter).second;

      if (isSupportedMachineType(MachineType::GPU))
      {
         funBody 
            << STR_GPU_CHECK_START;
         funBody << TAB << "mi" << miSN << ".getBlocks(blengths, blocs, ";
         if ((*it)->isPointer()) funBody << "*";
         funBody 
            << (*it)->getName(MachineType::GPU) << ");\n"
            << "#else\n";
      }
      funBody << TAB << "mi" << miSN << ".getBlocks(blengths, blocs, ";
      if ((*it)->isPointer()) funBody << "*";
      funBody << (*it)->getName() << ");\n";
      if (isSupportedMachineType(MachineType::GPU))
      {
         funBody 
            << STR_GPU_CHECK_END;
      }
   }
   initializeProxyGetSendType->setFunctionBody(funBody.str());
   instance.addMethod(initializeProxyGetSendType);
}
