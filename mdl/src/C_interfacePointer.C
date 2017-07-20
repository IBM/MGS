// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_interfacePointer.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "Interface.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <sstream>
#include <iostream>

void C_interfacePointer::execute(MdlContext* context) 
{
   if (_name == "") {
      throw InternalException("_name is empty in C_interfacePointer::execute");
   }
   try {
      Generatable* gen = context->_generatables->getMember(_name);
      _interface = dynamic_cast<Interface*>(gen);      
      if (_interface == 0) {
	 std::ostringstream stream;
	 stream << _name << " is found, but has a different type.";
	 throw NotFoundException(stream.str());	    
      }
   } catch (NotFoundException& e) {
      std::ostringstream os;
      os << "Problem retrieving " << _name << " as an Interface; " 
	 << e.getError();
      e.setError("");
      throw SyntaxErrorException(os.str());      
   }
}

C_interfacePointer::C_interfacePointer() 
   : C_production(), _interface(0), _name("") 
{

}

C_interfacePointer::C_interfacePointer(const std::string& name) 
   : C_production(), _interface(0), _name(name) 
{

}

C_interfacePointer::C_interfacePointer(const C_interfacePointer& rv) 
   : C_production(rv), _interface(rv._interface), _name(rv._name) 
{

}

void C_interfacePointer::duplicate(std::auto_ptr<C_interfacePointer>& rv) const
{
   rv.reset(new C_interfacePointer(*this));
}

Interface* C_interfacePointer::getInterface() 
{
   return _interface;
}

void C_interfacePointer::setInterface(Interface* interface) 
{
   _interface = interface;
}

const std::string& C_interfacePointer::getName() const
{
   return _name;
}

void C_interfacePointer::setName(const std::string& name) 
{
   _name = name;
}

C_interfacePointer::~C_interfacePointer() 
{

}


