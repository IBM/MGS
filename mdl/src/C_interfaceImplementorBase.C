// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_interfaceImplementorBase.h"
#include "C_generalList.h"
#include "C_instanceMapping.h"
#include "InterfaceImplementorBase.h"
#include "C_interfacePointerList.h"
#include "MdlContext.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "NotFoundException.h"
#include "Interface.h"
#include "StructType.h"
#include "MemberToInterface.h"
#include "Constants.h"
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <iostream>

void C_interfaceImplementorBase::execute(MdlContext* context) 
{
   // look at: void C_interfaceImplementorBase::
   // executeInterfaceImplementorBase(MdlContext* context, 
   // CompCategoryBase* cc) 
}

C_interfaceImplementorBase::C_interfaceImplementorBase() 
   : C_production(), _name(""), _interfacePointerList(0), _generalList(0) 
{
}

C_interfaceImplementorBase::C_interfaceImplementorBase(
   const std::string& name, C_interfacePointerList* ipl, C_generalList* gl) 
   : C_production(), _name(name), _interfacePointerList(ipl), _generalList(gl) 
{ 
}

C_interfaceImplementorBase::C_interfaceImplementorBase(
   const C_interfaceImplementorBase& rv) 
   : C_production(rv), _name(rv._name), _interfacePointerList(0), 
     _generalList(0)  
{
   if (rv._interfacePointerList) {
      std::unique_ptr<C_interfacePointerList> dup;
      rv._interfacePointerList->duplicate(std::move(dup));
      _interfacePointerList = dup.release();
   }
   if (rv._generalList) {
      std::unique_ptr<C_generalList> dup;
      rv._generalList->duplicate(std::move(dup));
      _generalList = dup.release();
   }
}

void C_interfaceImplementorBase::duplicate(
   std::unique_ptr<C_interfaceImplementorBase>&& rv) const
{
   rv.reset(new C_interfaceImplementorBase(*this));
}

void C_interfaceImplementorBase::executeInterfaceImplementorBase(
   MdlContext* context, InterfaceImplementorBase* cc) const
{
   if (_generalList == 0) {
      throw InternalException(
	 "_generalList is 0 in C_interfaceImplementorBase::executeCompCategoryBase");
   }
   try {
      _generalList->execute(context);
   } catch (DuplicateException& e) {
      std::cerr << "In " << _name << ", " << e.getError() << "." << std::endl;
      e.setError("");
      throw;
   }
   cc->setName(_name);
   if (_interfacePointerList) {
      _interfacePointerList->execute(context);
      std::unique_ptr<std::vector<Interface*> > interfaceVec;
      _interfacePointerList->releaseInterfaceVec(interfaceVec);
      std::vector<Interface*>::iterator it, end = interfaceVec->end();
      for (it = interfaceVec->begin(); it != end; it++) {
	 std::unique_ptr<MemberToInterface> mti;
	 mti.reset(new MemberToInterface(*it));
	 cc->addMemberToInterfaceMapping(std::move(mti));
      }
   }

   if (_generalList->getDataTypeVec()) {
      std::unique_ptr<std::vector<DataType*> > dataTypeVec;
      _generalList->releaseDataTypeVec(dataTypeVec);
      std::vector<DataType*>::iterator it, end = dataTypeVec->end();
      for (it = dataTypeVec->begin(); it != end; it++) {
	 std::unique_ptr<DataType> dataType;
	 dataType.reset(*it);
	 cc->addDataTypeToInstances(std::move(dataType));
      }
   }
   if (_generalList->getOptionalDataTypeVec()) {
      std::unique_ptr<std::vector<DataType*> > optionalDataTypeVec;
      _generalList->releaseOptionalDataTypeVec(optionalDataTypeVec);
      std::vector<DataType*>::iterator it, end = optionalDataTypeVec->end();
      for (it = optionalDataTypeVec->begin(); it != end; it++) {
	 std::unique_ptr<DataType> dataType;
	 dataType.reset(*it);
	 cc->addDataTypeToOptionalServices(std::move(dataType));
      }
   }
   if (_generalList->getInstanceMappingVec()) {
      std::unique_ptr<std::vector<C_instanceMapping*> > imVec;
      _generalList->releaseInstanceMappingVec(imVec);
      std::vector<C_instanceMapping*>::iterator it, end = imVec->end();
      for (it = imVec->begin(); it != end; it++) {
	 const DataType* curDt;
	 std::unique_ptr<DataType> dtToInsert;
	 try {
	    curDt = cc->getInstances().getMember((*it)->getMember());
	    curDt->duplicate(std::move(dtToInsert));
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::cerr << "In " << _name << " instance member " 
		      << e.getError() << std::endl;
	    e.setError("");
	    throw;
	 }
	 cc->addMappingToInterface((*it)->getInterface(), 
				   (*it)->getInterfaceMember(), std::move(dtToInsert),
				   (*it)->getAmpersand());
	 delete *it;
      }
   }
   std::unique_ptr<StructType> outAttr;
   if (_generalList->getOutAttrPSet()) {
      _generalList->releaseOutAttrPSet(std::move(outAttr));
   } else {
      outAttr.reset(new StructType());
   }
   outAttr->setName(OUTATTRPSETNAME);
   cc->setOutAttrPSet(std::move(outAttr));
   cc->setInterfaceImplementors();
}

C_interfaceImplementorBase::~C_interfaceImplementorBase() 
{
   delete _interfacePointerList;
   delete _generalList;
}


