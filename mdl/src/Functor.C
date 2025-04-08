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

#include "Functor.h"
#include "Generatable.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Class.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "Constants.h"
#include "VoidType.h"
#include "BaseClass.h"

#include <string>
#include <memory>
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdio.h>
#include <string.h>

Functor::Functor(const std::string& fileName)
   : ToolBase(fileName), _userExecute(false), _returnType(0), _category("")
{
   _executeArguments = new MemberContainer<DataType>();
}

Functor::Functor(const Functor& rv)
   : ToolBase(rv), _userExecute(rv._userExecute), _returnType(0)
   , _category(rv._category)
{
   copyOwnedHeap(rv);
}

Functor& Functor::operator=(const Functor& rv)
{
   if (this != &rv) {
      ToolBase::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _userExecute = rv._userExecute;
      _category = rv._category;
   }
   return *this;
}

void Functor::duplicate(std::unique_ptr<Generatable>&& rv) const
{
   rv.reset(new Functor(*this));
}

std::string Functor::getType() const
{
   return "Functor";
}

std::string Functor::generateExtra() const
{
   std::ostringstream os;
   os << "\t" << _returnType->getTypeString() << " Execute (";
   MemberContainer<DataType>::const_iterator it, next
      , end = _executeArguments->end();
   for (it = _executeArguments->begin(); it != end; it++) {
      os << it->second->getString();
      next = it;
      next++;
      if ( next != end) {
	 os << ", ";
      }
   }
   if (_userExecute) {
      if (_executeArguments->size()) {
	 os << ", ";
      }
      os << "...";
   }
   os << ");\n";
   return os.str();
}

std::string Functor::generateTitleExtra() const
{
   std::string retval = "";
   if (_category != "") {
      retval = "Category \"" + _category + "\" ";
   }
   return retval;
}

void Functor::setReturnType(std::unique_ptr<DataType>&& ret)
{
   _returnType = ret.release();
}

const std::string& Functor::getCategory() const
{
   return _category;
}

void Functor::setCategory(const std::string& category)
{
   _category = category;
}

Functor::~Functor()
{
   destructOwnedHeap();
}

std::string Functor::getTypeDescription()
{
   return _category;
}

void Functor::copyOwnedHeap(const Functor& rv)
{
   if (rv._executeArguments) {
      std::unique_ptr<MemberContainer<DataType> > dup;
      rv._executeArguments->duplicate(dup);
      _executeArguments = dup.release();
   } else {
      _executeArguments = 0;
   }
   if (rv._returnType) {
      std::unique_ptr<DataType> dup;
      rv._returnType->duplicate(std::move(dup));
      _returnType = dup.release();
   } else {
      _returnType = 0;
   }
}

void Functor::destructOwnedHeap()
{
   delete _executeArguments;
   delete _returnType;
}

std::string Functor::getModuleTypeName() const
{
   return "functor";
}

void Functor::internalGenerateFiles()
{
   assert(strcmp(getName().c_str(), ""));  
   generateInitArgs();
   generateExecArgs();
   generateInstanceBase();
   generateInstance();
   generateType();
   generateFactory();

}

void Functor::generateExecArgs() 
{
   generateInitializer("ExecArgs", *_executeArguments, _userExecute);
}

void Functor::generateInstanceBase()
{
   std::string fullName = PREFIX + getModuleName() + "Base";
   std::unique_ptr<Class> instance(new Class(fullName));
   
   std::unique_ptr<BaseClass> funcBase;
   if (getTypeDescription()=="LAYOUT") {
     funcBase.reset(new BaseClass("LayoutFunctor"));
     instance->addHeader("\"LayoutFunctor.h\"");
   }
   else if (getTypeDescription()=="NODEINITIALIZER") {
     funcBase.reset(new BaseClass("NodeInitializerFunctor"));
     instance->addHeader("\"NodeInitializerFunctor.h\"");
   }
   else if (getTypeDescription()=="CONNECTOR") {
     funcBase.reset(new BaseClass("ConnectorFunctor"));
     instance->addHeader("\"ConnectorFunctor.h\"");
   }
   else {
     funcBase.reset(new BaseClass("Functor"));
     instance->addHeader("\"Functor.h\"");
   }

   instance->addBaseClass(std::move(funcBase));
   instance->addHeader("\"DataItem.h\"");
   instance->addHeader("<memory>");

   instance->addDataTypeDataItemHeader(_returnType);
   instance->addDataTypeHeader(_returnType);
   
   std::unique_ptr<Attribute> initArgs(
      new CustomAttribute("init", PREFIX + getModuleName() + "InitArgs"));
   instance->addAttribute(initArgs);
   instance->addHeader("\"" + PREFIX + getModuleName() + "InitArgs.h\"");

   std::unique_ptr<Attribute> execArgs(
      new CustomAttribute("exec", PREFIX + getModuleName() + "ExecArgs"));
   instance->addAttribute(execArgs);
   instance->addHeader("\"" + PREFIX + getModuleName() + "ExecArgs.h\"");

   std::unique_ptr<Method> methodCup;
   createInitMethod(std::move(methodCup), _initializeArguments, "Initialize"
		    , "init", _userInitialization, false);
   instance->addMethod(std::move(methodCup));
   createInitMethod(std::move(methodCup), *_executeArguments, "Execute"
		    , "exec", _userExecute, true);
   instance->addMethod(std::move(methodCup));

   createUserMethod(std::move(methodCup), _initializeArguments, "Initialize"
		    , "void", _userInitialization, true);
   instance->addMethod(std::move(methodCup));
   std::string retStr;
   if (_returnType->isPointer() && _returnType->shouldBeOwned()) {
      retStr = "std::unique_ptr<" + _returnType->getDescriptor() + ">";
   } else {
      retStr = _returnType->getTypeString();
   }
   createUserMethod(std::move(methodCup), *_executeArguments, "Execute", retStr
		    , _userExecute, true);
   instance->addMethod(std::move(methodCup));

   instance->addStandardMethods();
   _classes.push_back(instance.release());
}

void Functor::createInitMethod(std::unique_ptr<Method>&& method, 
			       const MemberContainer<DataType>& args, 
			       const std::string& funcName, 
			       const std::string& attName, 
			       bool userInit, bool hasRetVal)
{
   method.reset(new Method("do" + funcName, "void") );
   method->addParameter("LensContext *c");
   method->addParameter("const std::vector<DataItem*>& args");
   method->setVirtual();
   method->setAccessType(AccessType::PROTECTED);
   std::ostringstream doInitFunctionBody;
   doInitFunctionBody 
      << TAB << ( userInit ? ("std::vector<DataItem*>::const_iterator " 
       + PREFIX + "currentDI = " ) : "" )
      << attName << ".initialize(args);\n"; 

   bool isRetValVoid = (dynamic_cast<VoidType*>(_returnType) != 0);
   if (hasRetVal) {
      method->addParameter("std::unique_ptr<DataItem>& rvalue");     
   }

   std::string userFunctionCaller =
      "user" + funcName + "(c";
   MemberContainer<DataType>::const_iterator it, end = args.end();
   for (it = args.begin(); it != end; it++) {     
      userFunctionCaller += ", " + attName + "." + it->second->getName();
   }   
   userFunctionCaller += 
      (userInit ? ", " + PREFIX + "currentDI, args.end()" : "") + ")";

   if (hasRetVal && !isRetValVoid) {
      if (_returnType->isPointer() && _returnType->shouldBeOwned()) {
	 doInitFunctionBody 
	    << TAB << "std::unique_ptr<" << _returnType->getDescriptor() 
	    << "> transfer(" << userFunctionCaller << ".release());\n"
	    << TAB << "rvalue.reset(new " << _returnType->getDataItemString() 
	    << "(transfer));\n";	 
      } else {
	 doInitFunctionBody 
	    << TAB << "rvalue.reset(new " << _returnType->getDataItemString() 
	    << "(" << userFunctionCaller << "));\n";
      }
   } else {
      doInitFunctionBody 
	 << TAB << userFunctionCaller << ";\n";
   }

   method->setFunctionBody(doInitFunctionBody.str());
}

void Functor::createUserMethod(std::unique_ptr<Method>&& method, 
			       const MemberContainer<DataType>& args, 
			       const std::string& funcName, 
			       const std::string& retType, 
			       bool userInit, bool pureVirtual)
{
   method.reset(new Method("user" + funcName, retType) );
   MemberContainer<DataType>::const_iterator it, end = args.end();
   method->addParameter("LensContext* " + PREFIX + "c");
   for (it = args.begin(); it != end; it++) {
      method->addParameter(it->second->getTypeString() + "& " + it->second->getName());
   }
   if (userInit) {
      method->addParameter("std::vector<DataItem*>::const_iterator begin");
      method->addParameter("std::vector<DataItem*>::const_iterator end");
   }
   if (pureVirtual) {
      method->setPureVirtual();
   }
}

void Functor::generateInstance()
{
   std::string moduleTypeNameCap = getModuleTypeName();
   moduleTypeNameCap[0] += 'A' - 'a';

   std::string baseName = PREFIX + getModuleName() + "Base";
   std::unique_ptr<Class> instance(new Class(getModuleName()));

   instance->setUserCode();

   instance->addDataTypeHeader(_returnType);

   std::unique_ptr<BaseClass> baseClass(new BaseClass(baseName));

   instance->addBaseClass(std::move(baseClass));
   instance->addHeader("\"" + baseName + ".h\"");
   instance->addHeader("\"LensContext.h\"");
   instance->addHeader("<memory>");
   
   std::unique_ptr<Method> methodCup;

   createUserMethod(std::move(methodCup), _initializeArguments, "Initialize"
		    , "void", _userInitialization, false);
   instance->addMethod(std::move(methodCup));
   std::string retStr;
   if (_returnType->isPointer() && _returnType->shouldBeOwned()) {
      retStr = "std::unique_ptr<" + _returnType->getDescriptor() + ">";
   } else {
      retStr = _returnType->getTypeString();
   }
   
   createUserMethod(std::move(std::move(methodCup)), *_executeArguments, "Execute", retStr
		    , _userExecute, false);
   instance->addMethod(std::move(methodCup));

   instance->addDuplicateType(moduleTypeNameCap);
   instance->addStandardMethods();
   _classes.push_back(instance.release());
}
