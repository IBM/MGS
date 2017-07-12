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

#include "C_interfaceMapping.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <cassert>
#include "C_identifierList.h"

// For now.
#include <iostream>


void C_interfaceMapping::execute(MdlContext* context) 
{
   _member->execute(context);
}

C_interfaceMapping::C_interfaceMapping() 
   : C_general(), _interface(""), _interfaceMember(""), _member(0) 
{
}

C_interfaceMapping::C_interfaceMapping(const std::string& interface,
				       const std::string& interfaceMember,
				       C_identifierList* member,
				       bool amp)
   : C_general(), _interface(interface), _interfaceMember(interfaceMember), 
     _member(member), _ampersand(amp)
{
} 

C_interfaceMapping::C_interfaceMapping(const C_interfaceMapping& rv)
   : _interface(rv._interface), _interfaceMember(rv._interfaceMember),
     _ampersand(rv._ampersand)
{
   copyOwnedHeap(rv);
}

C_interfaceMapping& C_interfaceMapping::operator=(const C_interfaceMapping& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _interface = rv._interface;
      _interfaceMember = rv._interfaceMember;
      _ampersand = rv._ampersand;
   }
   return *this;
}


void C_interfaceMapping::duplicate(std::auto_ptr<C_interfaceMapping>& rv) const
{
   rv.reset(new C_interfaceMapping(*this));
}

void C_interfaceMapping::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_interfaceMapping(*this));
}

C_interfaceMapping::~C_interfaceMapping() 
{
   destructOwnedHeap();
}

void C_interfaceMapping::destructOwnedHeap()
{
   delete _member;
}

void C_interfaceMapping::copyOwnedHeap(const C_interfaceMapping& rv)
{
   if (rv._member) {
      std::auto_ptr<C_identifierList> dup;
      rv._member->duplicate(dup);
      _member = dup.release();
   } else {
      _member = 0;
   }
}

const std::string& C_interfaceMapping::getMember() const {
   const std::vector<std::string>& ids = _member->getIdentifiers();

//    std::cout << _interface << " " << _interfaceMember 
// 	     << " The size is: " << ids.size() << std::endl;

   assert(ids.size() > 0);
   return ids[0];
}

bool C_interfaceMapping::getSubAttributePathExists() const
{
   return _member->getIdentifiers().size() > 1;
}

std::vector<std::string> C_interfaceMapping::getSubAttributePath() const
{
   const std::vector<std::string>& ids = _member->getIdentifiers();
   if (ids.size() <= 1) {
      return std::vector<std::string>();
   } else {
      return std::vector<std::string>(ids.begin()+1, ids.end());
   }
}
