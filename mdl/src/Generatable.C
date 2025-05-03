// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Generatable.h"
#include "InternalException.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include "BaseClass.h"
#include "DataType.h"
#include "MemberContainer.h"
#include "CommandLine.h"

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

Generatable::Generatable(const std::string& fileName)
   : _fileOutput(true), _fileName(fileName), _linkType(_DYNAMIC), 
     _frameWorkElement(false), _supportedMachineType(MachineType::CPU)
     , _cmdLine(0)
{
   //_supportedMachineTypes.insert(MachineType::CPU);
}

Generatable::Generatable(const Generatable& rv)
   : _fileOutput(rv._fileOutput), _fileName(rv._fileName), 
     _linkType(rv._linkType), _frameWorkElement(rv._frameWorkElement),
     //_supportedMachineTypes(rv._supportedMachineTypes)     
     _supportedMachineType(rv._supportedMachineType)     
     , _cmdLine(rv._cmdLine)
{
   copyOwnedHeap(rv);
}

Generatable& Generatable::operator=(const Generatable& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _fileOutput = rv._fileOutput;
      _fileName = rv._fileName;
      _linkType = rv._linkType;
      _frameWorkElement = rv._frameWorkElement;
      _cmdLine = rv._cmdLine;
   }
   return *this;
}

void Generatable::generateFiles(const std::string& originalFileName) 
{
   std::string modName = "\"" + originalFileName + "\"";
   if (_cmdLine && _cmdLine->skipIncludes()) {
      _fileOutput = modName == _fileName;  // Only generate the main file
   } else {
      _fileOutput = true;  // Generate all files including includes
   }
   
   internalGenerateFiles();
   createDirectoryStructure();
   generateModuleMk(); // hack for MBL
   
   for (std::vector<Class*>::iterator it = _classes.begin(); it != _classes.end(); it++) {
     (*it)->setFileOutput(_fileOutput);
     (*it)->generate(getModuleName());
   }
}

Generatable::~Generatable() 
{
   destructOwnedHeap();
}

std::string Generatable::getTypeDescription()
{
   return "";
}

void Generatable::copyOwnedHeap(const Generatable& rv)
{
   std::vector<Class*>::const_iterator it, end = rv._classes.end();
   for (it = rv._classes.begin(); it != end; it++) {
      std::unique_ptr<Class> dup;
      (*it)->duplicate(std::move(dup));
      _classes.push_back(dup.release());
   }
}

void Generatable::destructOwnedHeap()
{
   std::vector<Class*>::iterator it, end = _classes.end();
   for (it = _classes.begin(); it != end; it++) {
      delete *it;
   }
   _classes.clear();
}

void Generatable::generateModuleMk() 
{
   if (_frameWorkElement) {
      _linkType = _STATIC;
   }

   std::ostringstream os;

   os << "# The pathname is relative to the MGS directory" << "\n"
      << "THIS_DIR := " << "extensions/" << getModuleTypeName() 
      << "/" << getModuleName() << "\n"
      << "THIS_STEM := " << getModuleName() << "\n"
      << "\n"
      << "SRC_PREFIX := $(THIS_DIR)/src" << "\n"
      << "OBJ_PREFIX := $(THIS_DIR)/obj" << "\n"
      << "\n";

   os << "SOURCES := ";
   
   if (_classes.size() > 0) {
      std::vector<Class*>::iterator end = _classes.end();
      std::vector<Class*>::iterator it;
      for (it = _classes.begin(); it != end;) {
	      if ( (*it)->generateSourceFile() ) {
	         os << (*it)->getFileName() << ".C ";
	         it++;
	         if (it != end) {
	            os << "\\\n";
	         } else {
	            os << "\n";
	         }
	      }
	      else it++;
      }
   } else {
      throw InternalException(
	 "_files.size() > 0 is false in Generatable::generateModuleMk");
   }

   os << "\n"
      << "# define the full pathname for each file\n"
      << "SRC_$(THIS_STEM) = $(patsubst %,$(SRC_PREFIX)/%, $(SOURCES))\n"
      << "\n"
      << "THIS_SRC := $(SRC_$(THIS_STEM))\n"
      << "SRC += $(THIS_SRC)\n"
      << "\n"
      << "# Create the list of object files by substituting .C with .o\n"
      << "TEMP :=  $(patsubst %.C,%.o,$(filter %.C,$(THIS_SRC)))\n"
      << "\n"
      << "OBJ_$(THIS_STEM) := $(subst src,obj,$(TEMP))\n"
      << "OBJS += $(OBJ_$(THIS_STEM))\n"
      << "\n";

   if (_linkType == _DYNAMIC) {
      os
	 << "#Defined and Undefined symbols files\n"
	 << "$(SO_DIR)/" << getModuleName() << ".def: $(OBJ_" 
	 << getModuleName() << ")\n" // from prev line
	 << "	echo \\#\\!$(SO_DIR)/" << getModuleName() << ".so > $@\n"
	 << "	$(SCRIPTS_DIR)/gen_def.sh $^ >> $@\n"
	 << "$(SO_DIR)/" << getModuleName() << ".undef: $(OBJ_" 
	 << getModuleName() << ")\n" // from prev line
	 << "	echo \\#\\!$(SO_DIR)/" << getModuleName() << ".so > $@\n"
	 << "	$(SCRIPTS_DIR)/gen_undef.sh $^ >> $@\n"
	 << "\n"
	 << "DEF_SYMBOLS += $(SO_DIR)/" << getModuleName() << ".def\n"
	 << "UNDEF_SYMBOLS += $(SO_DIR)/" << getModuleName() << ".undef\n"
	 << "" << "\n"
	 << "#Shared Objects of this module" << "\n"
	 << "$(SO_DIR)/" << getModuleName() // conts next
	 << ".so: $(SO_DIR)/Dependfile $(OBJ_" << getModuleName() << ")\n"
	 << "	$(SHAREDCC)" << "\n"
	 << "\n"
	 << "GENERATED_DL_OBJECTS += $(OBJ_$(THIS_STEM))\n"
	 << "\n"
	 << "#Add the shared objects of this module to the general target\n"
	 << "SHARED_OBJECTS += $(SO_DIR)/" << getModuleName() << ".so\n"
	 << "\n"
	 << "MAINS := $(patsubst %,$(SRC_PREFIX)/%, $(MAINS))\n"
	 << "\n"
	 << "TEMP := $(patsubst %.C,%.o,$(MAINS))\n";
   } else { // _BASE
      os 
	 << "EXTENSION_OBJECTS += $(OBJ_$(THIS_STEM))\n";
   }
   
   if (_fileOutput) {
      std::string fName = getModuleName() + "/" + "module.mk";
      std::ofstream fs(fName.c_str());
      fs << os.str();
      fs.close();
   } 
}

void Generatable::createDirectoryStructure()
{
   if (_fileOutput) {
      std::string sysCall = "mkdir -p " + getModuleName() + " ; " 
	 + "mkdir -p " + getModuleName() + "/src ; " 
	 + "mkdir -p " + getModuleName() + "/include ; ";
	 //+ "mkdir -p " + getModuleName() + "/obj ; "; //TUAN (09-25-2018): there is no need for this, as all object files are put into the same folder
      try {
	 system(sysCall.c_str());
      } catch(...) {};
   } 
}

void Generatable::generateType()
{
   std::string moduleTypeNameCap = getModuleTypeName();
   moduleTypeNameCap[0] += 'A' - 'a';
   std::string moduleTypeNameCapType = moduleTypeNameCap + "Type";
   std::string fullName = PREFIX + getModuleName() + "Type";
   std::unique_ptr<Class> instance(new Class(fullName));

   std::string instanceInitializer = 
      "new " + getInstanceNameForType() + "(" + 
      getInstanceNameForTypeArguments() + ")";

   std::unique_ptr<BaseClass> baseClass(new BaseClass(moduleTypeNameCapType));
   
   addGenerateTypeClassAttribute(*(instance.get()));

   instance->addBaseClass(std::move(baseClass));
   instance->addHeader("\"" + moduleTypeNameCapType + ".h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<string>");
   instance->addExtraSourceHeader("\"" + getInstanceNameForType() + ".h\"");
   instance->addExtraSourceHeader("\"InstanceFactoryQueriable.h\"");
   instance->addClass(moduleTypeNameCap);

   std::unique_ptr<Method> getGeneratableAutoMethod(
      new Method("get" + moduleTypeNameCap, "void") );
   getGeneratableAutoMethod->setVirtual(true);
   getGeneratableAutoMethod->addParameter(
      "std::unique_ptr<" + moduleTypeNameCap + ">& aptr");
   getGeneratableAutoMethod->setFunctionBody(
      TAB + "aptr.reset(" + instanceInitializer +");\n");
   instance->addMethod(std::move(getGeneratableAutoMethod));

   std::unique_ptr<Method> getGeneratableMethod(
      new Method("get" + moduleTypeNameCap, moduleTypeNameCap + "*") );
   getGeneratableMethod->setVirtual(true);
   std::ostringstream getGeneratableMethodFunctionBody;
   getGeneratableMethodFunctionBody 
      << TAB << moduleTypeNameCap << "* s = " << instanceInitializer << ";\n"
      << TAB << "_" << getModuleTypeName() << "List.push_back(s);\n"
      << TAB << "return s;\n";
   getGeneratableMethod->setFunctionBody(
      getGeneratableMethodFunctionBody.str());
   instance->addMethod(std::move(getGeneratableMethod));

   std::unique_ptr<Method> getNameMethod(
      new Method("getName", "std::string",
		 TAB + "return \"" + getModuleName() + "\";\n") );
   getNameMethod->setVirtual(true);
   instance->addMethod(std::move(getNameMethod));

   std::unique_ptr<Method> getDescriptionMethod(
      new Method("getDescription", "std::string", TAB + "return \"" 
		 + getTypeDescription() + "\";\n") );
   getDescriptionMethod->setVirtual(true);
   instance->addMethod(std::move(getDescriptionMethod));

   std::unique_ptr<Method> getQueriableMethod(
      new Method("getQueriable", "void"));
   getQueriableMethod->setVirtual(true);
   getQueriableMethod->addParameter(
      "std::unique_ptr<InstanceFactoryQueriable>& dup");
   std::ostringstream getQueriableMethodFunctionBody;
   getQueriableMethodFunctionBody 
      << TAB << "dup.reset(new InstanceFactoryQueriable(this));\n"
      << TAB << "dup->setName(getName());\n";
   getQueriableMethod->setFunctionBody(getQueriableMethodFunctionBody.str());
   instance->addMethod(std::move(getQueriableMethod));
   
   instance->addStandardMethods();
   _classes.push_back(instance.release());
}

void Generatable::generateFactory()
{
   std::string moduleTypeNameCap = getModuleTypeName();
   moduleTypeNameCap[0] += 'A' - 'a';

   std::string loadedInstanceTypeName = getLoadedInstanceTypeName();
   std::string loadedBaseTypeName = moduleTypeNameCap + "Type";

   std::string fullName = PREFIX + getModuleName() + "Factory";
   std::unique_ptr<Class> instance(new Class(fullName));

   instance->addClass(loadedBaseTypeName);
   instance->addClass("NDPairList");
   instance->addClass("Simulation");
   instance->addExtraSourceHeader("\"" + loadedInstanceTypeName + ".h\"");
   instance->addExtraSourceHeader("\"FactoryMap.h\"");

   std::unique_ptr<Method> constructor(new Method(fullName));
   constructor->setFunctionBody(TAB + "FactoryMap<" + loadedBaseTypeName +
				">::getFactoryMap()->addFactory(\""
				+ getModuleName() + "\", " 
				+ PREFIX + getModuleName() 
				+ "FactoryFunction);\n");

   std::unique_ptr<Method> factoryFunction(
      new Method(fullName + "Function", loadedBaseTypeName 
						    + "*"));   
   factoryFunction->addParameter("Simulation& s");
   factoryFunction->addParameter("const NDPairList& ndpList");
   factoryFunction->setExternC(true);
   factoryFunction->setFunctionBody(
      TAB + TAB + "return new " + loadedInstanceTypeName + "(" 
      + getLoadedInstanceTypeArguments() + ");\n");

   std::unique_ptr<Method> destructor(new Method("~" + fullName));

   instance->addMethod(std::move(factoryFunction));
   instance->addMethod(std::move(constructor));
   instance->addMethod(std::move(destructor));

// Disabled for now for strange AIX behavior with gcc 3.3.3
//   instance->addExtraSourceString(fullName + " " + fullName + ";\n");
// Used for now for strange AIX behavior with gcc 3.3.3
   instance->setSourceFileBeginning("#include \"" + fullName + ".h\"\n" +
				    fullName + " " + fullName + ";\n");

   _classes.push_back(instance.release());
}

std::string Generatable::getLoadedInstanceTypeName()
{
   return PREFIX + getModuleName() + "Type";
}

std::string Generatable::getLoadedInstanceTypeArguments()
{
   return "";
}

void Generatable::addDoInitializeMethods(
   Class& instance, const MemberContainer<DataType>& members) const
{
   instance.addExtraSourceHeader("\"SyntaxErrorException.h\"");
   instance.addExtraSourceHeader("<sstream>");
   std::unique_ptr<Method> doInitMethod(new Method("doInitialize", "void") );
   doInitMethod->addParameter("GslContext *c");
   doInitMethod->addParameter("const std::vector<DataItem*>& args");
   doInitMethod->setVirtual();
   doInitMethod->setAccessType(AccessType::PROTECTED);
   doInitMethod->setFunctionBody(getDoInitializeMethodBody(members));
   instance.addMethod(std::move(doInitMethod));

   instance.addExtraSourceHeader("\"NDPairList.h\"");
   std::unique_ptr<Method> setUpMethod(new Method("doInitialize", "void") );
   setUpMethod->addParameter("const NDPairList& ndplist");
   setUpMethod->setVirtual();
   setUpMethod->setAccessType(AccessType::PROTECTED);
   setUpMethod->setFunctionBody(getSetupFromNDPairListMethodBody(members));
   instance.addMethod(std::move(setUpMethod));
}

std::string Generatable::getDoInitializeMethodBody(
   const MemberContainer<DataType>& members) const
{
   if (members.size() == 0) {
      return "";
   }

   std::ostringstream doInitFunctionBody;
   std::ostringstream doInitFunctionSubBody;
   doInitFunctionSubBody 
      << TAB << "std::vector<DataItem*>::const_iterator " 
      + PREFIX + "currentDI = args.begin();\n";
   int initSize = 0;

   MemberContainer<DataType>::const_iterator it, end = members.end();
   std::string initStr;
   for (it = members.begin(); it != end; it++) {
      initStr = it->second->getInitializerString(PREFIX + "currentDI");
      if (initStr != "") {
	 doInitFunctionSubBody << initStr;
	 ++initSize;
      }
   }

   doInitFunctionBody 
      << TAB << "if (args.size() != " << initSize << ") {\n"
      << TAB << TAB << "std::ostringstream " << PREFIX << "mes;\n"
      << TAB << TAB << PREFIX << "mes << \"In " << getModuleName() 
      << " the incoming args size is \" << args.size() << \" but \" << " 
      << initSize<< " << \" is expected.\";\n"
      << TAB << TAB << "throw SyntaxErrorException(" 
      << PREFIX << "mes.str());\n"
      << TAB << "}\n" << doInitFunctionSubBody.str();
   return doInitFunctionBody.str();
}

std::string Generatable::getSetupFromNDPairListMethodBody(
   const MemberContainer<DataType>& members) const
{
   if (members.size() == 0) {
      return "";
   }

   std::ostringstream os;
   os << TAB << "NDPairList::const_iterator it, end = ndplist.end();\n"
      << TAB << "for (it = ndplist.begin(); it != end; it++) {\n"
      << TAB << TAB << "bool " << FOUND << " = false;\n";
   bool first = true;

   MemberContainer<DataType>::const_iterator it, end = members.end();
   for (it = members.begin(); it != end; it++) {
      os << it->second->getPSetString("(*it)", first);
      if (first) {
	 first = false;
      }
   }

   os << TAB << TAB << "if (!" << FOUND << ") {\n"
      << TAB << TAB << TAB << "std::ostringstream os;\n"
      << TAB << TAB << TAB << "os << (*it)->getName() << "
      << "\" can not be handled in \" << typeid(*this).name();\n"
      << TAB << TAB << TAB << "os << \" HINTS: the data member name is not available but you may be using it somewhere (e.g. in GSL file or the parameter file)\";\n"
      << TAB << TAB << TAB << "throw SyntaxErrorException(os.str());\n"
      << TAB << TAB << "}\n"      
      << TAB << "}\n";
   return os.str();
}

void Generatable::addSelfToExtensionModules(
   std::map<std::string, std::vector<std::string> >& modules)
{
   if (!_frameWorkElement) {
      modules[getModuleTypeName()].push_back(getModuleName());
   }
}

void Generatable::addSelfToCopyModules(
   std::map<std::string, std::vector<std::string> >& modules)
{
   if ((!_frameWorkElement) && _fileOutput) {
      modules[getModuleTypeName()].push_back(getModuleName());
   }
}
