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
      std::auto_ptr<C_interfacePointerList> dup;
      rv._interfacePointerList->duplicate(dup);
      _interfacePointerList = dup.release();
   }
   if (rv._generalList) {
      std::auto_ptr<C_generalList> dup;
      rv._generalList->duplicate(dup);
      _generalList = dup.release();
   }
}

void C_interfaceImplementorBase::duplicate(
   std::auto_ptr<C_interfaceImplementorBase>& rv) const
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
      std::auto_ptr<std::vector<Interface*> > interfaceVec;
      _interfacePointerList->releaseInterfaceVec(interfaceVec);
      std::vector<Interface*>::iterator it, end = interfaceVec->end();
      for (it = interfaceVec->begin(); it != end; it++) {
	 std::auto_ptr<MemberToInterface> mti;
	 mti.reset(new MemberToInterface(*it));
	 cc->addMemberToInterfaceMapping(mti);
      }
   }

   if (_generalList->getDataTypeVec()) {
      std::auto_ptr<std::vector<DataType*> > dataTypeVec;
      _generalList->releaseDataTypeVec(dataTypeVec);
      std::vector<DataType*>::iterator it, end = dataTypeVec->end();
      for (it = dataTypeVec->begin(); it != end; it++) {
	 std::auto_ptr<DataType> dataType;
	 dataType.reset(*it);
	 cc->addDataTypeToInstances(dataType);
      }
   }
   if (_generalList->getOptionalDataTypeVec()) {
      std::auto_ptr<std::vector<DataType*> > optionalDataTypeVec;
      _generalList->releaseOptionalDataTypeVec(optionalDataTypeVec);
      std::vector<DataType*>::iterator it, end = optionalDataTypeVec->end();
      for (it = optionalDataTypeVec->begin(); it != end; it++) {
	 std::auto_ptr<DataType> dataType;
	 dataType.reset(*it);
	 cc->addDataTypeToOptionalServices(dataType);
      }
   }
   if (_generalList->getInstanceMappingVec()) {
      std::auto_ptr<std::vector<C_instanceMapping*> > imVec;
      _generalList->releaseInstanceMappingVec(imVec);
      std::vector<C_instanceMapping*>::iterator it, end = imVec->end();
      for (it = imVec->begin(); it != end; it++) {
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;
	 try {
	    curDt = cc->getInstances().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
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
				   (*it)->getInterfaceMember(), dtToInsert,
				   (*it)->getAmpersand());
	 delete *it;
      }
   }
   std::auto_ptr<StructType> outAttr;
   if (_generalList->getOutAttrPSet()) {
      _generalList->releaseOutAttrPSet(outAttr);
   } else {
      outAttr.reset(new StructType());
   }
   outAttr->setName(OUTATTRPSETNAME);
   cc->setOutAttrPSet(outAttr);
   cc->setInterfaceImplementors();
}

C_interfaceImplementorBase::~C_interfaceImplementorBase() 
{
   delete _interfacePointerList;
   delete _generalList;
}


