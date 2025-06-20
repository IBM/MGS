// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_struct.h"
#include "MdlContext.h"
#include "C_dataTypeList.h"
#include "StructType.h"
#include "DataType.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "C_general.h"
#include "C_generalList.h"
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>

void C_struct::execute(MdlContext* context) 
{
   bool hasName = true;

   if (_name == "") { // InAttrPSet
      hasName = false;
      //throw InternalException("_name is empty in C_struct::execute");
   }
   if (_dataTypeList == 0) {
      throw InternalException("_dataTypeList is 0 in C_struct::execute");
   }
   _struct = new StructType(getFileName());
   _struct->setFrameWorkElement(_frameWorkElement);
   if (hasName) {
      _struct->setTypeName(_name);
   }
   _dataTypeList->execute(context);
   std::unique_ptr<std::vector<DataType*> > dtv;
   _dataTypeList->releaseDataTypeVec(dtv);
   if (dtv.get()) {
      std::vector<DataType*>::reverse_iterator end = dtv->rend();
      std::vector<DataType*>::reverse_iterator it;
      for (it = dtv->rbegin(); it != end; it++) {
	 try {
	    std::unique_ptr<DataType> member;
	    member.reset(*it);
	    _struct->_members.addMemberToFront(member->getName(), std::move(member));
	 } catch (DuplicateException& e) {
	    std::cerr << e.getError() << std::endl;
	    e.setError("");
	 }
      }
   } else {
      throw InternalException("dtv is 0 in C_struct::execute");
   }
   if (hasName) {
      std::unique_ptr<Generatable> structMember;
      structMember.reset(_struct);
      _struct = 0;
      context->_generatables->addMember(_name, structMember);
   }
}

void C_struct::addToList(C_generalList* gl)
{
   assert(_struct != 0);
}


C_struct::C_struct() 
   : C_general(), _name(""), _struct(0), _dataTypeList(0), 
     _frameWorkElement(false)
{

}

C_struct::C_struct(C_dataTypeList* dtl) 
   : C_general(), _name(""), _struct(0), _dataTypeList(dtl),
     _frameWorkElement(false)
{

}

C_struct::C_struct(const std::string& name, C_dataTypeList* dtl, 
		   bool frameWorkElement) 
   : C_general(), _name(name), _struct(0), _dataTypeList(dtl), 
     _frameWorkElement(frameWorkElement) 
{

}

C_struct::C_struct(const C_struct& rv) 
   : C_general(rv), _name(rv._name), _struct(0), _dataTypeList(0),
     _frameWorkElement(rv._frameWorkElement) 
{
   if (rv._struct) {
      std::unique_ptr<DataType> dup;
      rv._struct->duplicate(std::move(dup));
      _struct = dynamic_cast<StructType*>(dup.release());
   }
   if (rv._dataTypeList) {
      std::unique_ptr<C_dataTypeList> dup;
      rv._dataTypeList->duplicate(std::move(dup));
      _dataTypeList = dup.release();
   }
}

void C_struct::duplicate(std::unique_ptr<C_struct>&& rv) const
{
   rv.reset(new C_struct(*this));
}

void C_struct::duplicate(std::unique_ptr<C_general>&& rv)const
{
   rv.reset(new C_struct(*this));
}


C_struct::~C_struct() 
{
   delete _struct;
   delete _dataTypeList;
}


