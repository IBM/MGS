// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
		if (it->second->isArray()) {
			//to support new CPU-GPU design
			//create 2 methods [one used in HAVE_GPU and one is not]
			{
				//#if defined(HAVE_GPU)
				// convert ShallowArray< double >* 
				// into 
				// ShallowArray_Flat< double, Array_Flat<int>::MemLocation::UNIFIED_MEM>*
				std::string  type = it->second->getTypeString(); 
				std::string from = "ShallowArray<";
				std::string to = "ShallowArray_Flat<";
				type = type.replace(type.find(from),from.length(),to);
				std::size_t start = type.find_first_of("<");
				std::size_t last = type.find_first_of(">");
				std::string element_datatype = type.substr(start+1, last-start-1);
				type = type.replace(start+1, last-start-1, element_datatype + ", " + MEMORY_LOCATION);

				std::unique_ptr<Method> producer(
				new Method(PREFIX + "get_" + _name + "_" + it->second->getName()
					, type));

				MacroConditional gpuConditional(GPUCONDITIONAL);
				producer->setMacroConditional(gpuConditional);
				producer->setPureVirtual();
				c.addMethod(std::move(producer));

			}
			{
				std::unique_ptr<Method> producer(
				new Method(PREFIX + "get_" + _name + "_" + it->second->getName()
					, it->second->getTypeString()));

				MacroConditional gpuConditional(GPUCONDITIONAL);
				gpuConditional.setNegateCondition();
				producer->setMacroConditional(gpuConditional);
				producer->setPureVirtual();
				c.addMethod(std::move(producer));
			}
		}
		else {
			std::unique_ptr<Method> producer(
				new Method(PREFIX + "get_" + _name + "_" + it->second->getName()
				, it->second->getTypeString()));
			producer->setPureVirtual();
			c.addMethod(std::move(producer));
		}
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

void Interface::addDataTypeToMembers(std::unique_ptr<DataType>&& dataType) 
{
   _members.addMemberToFront(dataType->getName(), std::move(dataType));
}
