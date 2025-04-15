#include <memory>
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BaseClass.h"
#include "Attribute.h"

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

BaseClass::BaseClass(const std::string& name, const std::string& conditional)
   : _name(name), _conditional(conditional)
{
}

BaseClass::BaseClass(const std::string& name)
   : _name(name), _conditional("")
{
}

BaseClass::BaseClass(const BaseClass& rv)
   : _name(rv._name), _conditional(rv._conditional)
{
   copyOwnedHeap(rv);
}

BaseClass& BaseClass::operator=(const BaseClass& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _name = rv._name;
      _conditional = rv._conditional;
   }
   return *this;
}

void BaseClass::duplicate(std::unique_ptr<BaseClass>&& dup) const
{
   dup.reset(new BaseClass(*this));
}

BaseClass::~BaseClass()
{
   destructOwnedHeap();
}

const std::string& BaseClass::getName() const 
{
   return _name;
}

const std::string& BaseClass::getConditional() const 
{
   return _conditional;
}

void BaseClass::setName(const std::string name)
{
  _name = name;
}

const std::vector<Attribute*>& BaseClass::getAttributes() const
{
   return _attributes;
}

void BaseClass::addAttribute(std::unique_ptr<Attribute>&& att)
{
   _attributes.push_back(att.release());
}
     
std::string BaseClass::getInitString() const
{
   std::ostringstream retVal;
   bool first = true;
   for (std::vector<Attribute*>::const_iterator it = _attributes.begin(); it != _attributes.end(); it++) {
      if (first) {
	 first = false;
      } else {
	retVal << ", ";
      }
      retVal << (*it)->getConstructorParameterNameExtra();
   } 
   return retVal.str();
}


void BaseClass::destructOwnedHeap()
{
   for (std::vector<Attribute*>::iterator it = _attributes.begin();
	it != _attributes.end(); it++) {
      delete *it;
   }
   _attributes.clear();
}

void BaseClass::copyOwnedHeap(const BaseClass& rv)
{
   for (std::vector<Attribute*>::const_iterator it = rv._attributes.begin();
	it != rv._attributes.end(); it++) {
      std::unique_ptr<Attribute> dup;
      (*it)->duplicate(std::move(dup));
      _attributes.push_back(dup.release());
   }
}

