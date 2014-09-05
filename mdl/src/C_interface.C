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

#include "C_interface.h"
#include "MdlContext.h"
#include "C_dataTypeList.h"
#include "DataType.h"
#include "Interface.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <vector>
#include <iostream>

void C_interface::execute(MdlContext* context) 
{
   if (_name == "") {
      throw InternalException("_name is empty in C_interface::execute");
   }
   if (_dataTypeList == 0) {
      throw InternalException("_dataTypeList is 0 in C_interface::execute");
   }
   _interface = new Interface(getFileName());
   _interface->setName(_name);
   _dataTypeList->execute(context);
   std::auto_ptr<std::vector<DataType*> > dtv;
   _dataTypeList->releaseDataTypeVec(dtv);
   if (dtv.get()) {
      std::vector<DataType*>::reverse_iterator end = dtv->rend();
      std::vector<DataType*>::reverse_iterator it;
      for (it = dtv->rbegin(); it != end; it++) {
	 try {
	    // Check if suitable
	    if (!(*it)->isSuitableForInterface()) {
	       throw SyntaxErrorException(
		  (*it)->getName() + 
		  " is not suitable to be in an interface.");
	    }
	    // insert
	    std::auto_ptr<DataType> member;
	    member.reset(*it);
	    _interface->addDataTypeToMembers(member);
	 } catch (DuplicateException& e) {
	    std::cerr << e.getError() << std::endl;
	    e.setError("");
	 }
      }
   } else {
      throw InternalException("dtv is 0 in C_interface::execute");
   }
   std::auto_ptr<Generatable> interfaceMember;
   interfaceMember.reset(_interface);
   _interface = 0;
   context->_generatables->addMember(_name, interfaceMember);

}

C_interface::C_interface() 
   : C_production(), _name(""), _interface(0), _dataTypeList(0) 
{

}

C_interface::C_interface(const std::string& name, C_dataTypeList* dtl) 
   : C_production(), _name(name), _interface(0), _dataTypeList(dtl) 
{

}

C_interface::C_interface(const C_interface& rv) 
   : C_production(rv), _name(rv._name), _interface(0), _dataTypeList(0) 
{
   if (rv._interface) {
      std::auto_ptr<Interface> dup;
      rv._interface->duplicate(dup);
      _interface = dup.release();
   }
   if (rv._dataTypeList) {
      std::auto_ptr<C_dataTypeList> dup;
      rv._dataTypeList->duplicate(dup);
      _dataTypeList = dup.release();
   }
}

void C_interface::duplicate(std::auto_ptr<C_interface>& rv) const
{
   rv.reset(new C_interface(*this));
}

C_interface::~C_interface() 
{
   delete _interface;
   delete _dataTypeList;
}


