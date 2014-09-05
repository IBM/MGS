// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
   std::auto_ptr<std::vector<DataType*> > dtv;
   _dataTypeList->releaseDataTypeVec(dtv);
   if (dtv.get()) {
      std::vector<DataType*>::reverse_iterator end = dtv->rend();
      std::vector<DataType*>::reverse_iterator it;
      for (it = dtv->rbegin(); it != end; it++) {
	 try {
	    std::auto_ptr<DataType> member;
	    member.reset(*it);
	    _struct->_members.addMemberToFront(member->getName(), member);
	 } catch (DuplicateException& e) {
	    std::cerr << e.getError() << std::endl;
	    e.setError("");
	 }
      }
   } else {
      throw InternalException("dtv is 0 in C_struct::execute");
   }
   if (hasName) {
      std::auto_ptr<Generatable> structMember;
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
      std::auto_ptr<DataType> dup;
      rv._struct->duplicate(dup);
      _struct = dynamic_cast<StructType*>(dup.release());
   }
   if (rv._dataTypeList) {
      std::auto_ptr<C_dataTypeList> dup;
      rv._dataTypeList->duplicate(dup);
      _dataTypeList = dup.release();
   }
}

void C_struct::duplicate(std::auto_ptr<C_struct>& rv) const
{
   rv.reset(new C_struct(*this));
}

void C_struct::duplicate(std::auto_ptr<C_general>& rv)const
{
   rv.reset(new C_struct(*this));
}


C_struct::~C_struct() 
{
   delete _struct;
   delete _dataTypeList;
}


