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

#include "Interface.h"
#include "DataType.h"
#include "Generatable.h"
#include "MemberContainer.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"

#include <string>
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>

Interface::Interface(const std::string& fileName) 
   : Generatable(fileName), _name("") 
{
}

void Interface::duplicate(std::unique_ptr<Generatable>&& rv) const
{
   rv.reset(new Interface(*this));
}

void Interface::duplicate(std::unique_ptr<Interface>&& rv) const
{
   rv.reset(new Interface(*this));
}

void Interface::generate() const
{
   std::cout << "Interface " << _name << " { " << std::endl;
   MemberContainer<DataType>::const_iterator end = _members.end();
   MemberContainer<DataType>::const_iterator it;
   for (it = _members.begin(); it != end; it ++) {
      std::cout << "\t" << it->second->getString() << ";" << std::endl;
   }
   std::cout << "}\n" << std::endl;
}

const std::string& Interface::getName() const
{
   return _name;
}

void Interface::setName(const std::string& name) 
{
   _name = name;
}

void Interface::addProducerMethods(Class& c)
{
   MemberContainer<DataType>::const_iterator end = _members.end();
   MemberContainer<DataType>::const_iterator it;
   for (it = _members.begin(); it != end; it ++) {
      std::unique_ptr<Method> producer(
	 new Method(PREFIX + "get_" + _name + "_" + it->second->getName()
		    , it->second->getTypeString()));
      producer->setPureVirtual();
      c.addMethod(std::move(producer));
   }
}


Interface::~Interface() 
{
}

std::string Interface::getModuleName() const
{
   return getName();
}

std::string Interface::getModuleTypeName() const
{
   return "interface";
}

void Interface::internalGenerateFiles() 
{
   assert(strcmp(getName().c_str(),""));
   generateInstance();
}

void Interface::generateInstance() 
{
   std::unique_ptr<Class> instance(new Class(getName()));
   instance->addDataTypeHeaders(_members);
   instance->addBasicInlineDestructor(true);
   addProducerMethods(*instance);
   _classes.push_back(instance.release());
}

void Interface::addDataTypeToMembers(std::unique_ptr<DataType>&& dataType) {
   _members.addMemberToFront(dataType->getName(), std::move(dataType));
}
