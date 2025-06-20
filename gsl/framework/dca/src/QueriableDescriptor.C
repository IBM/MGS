// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "QueriableDescriptor.h"
#include "QueryField.h"

QueriableDescriptor::QueriableDescriptor()
: _ptrQueriable(0)
{
}


QueriableDescriptor::QueriableDescriptor(QueriableDescriptor& qd)
: _name(qd._name), _description(qd._description), _type(qd._type), _ptrQueriable(qd._ptrQueriable)
{
}


QueriableDescriptor& QueriableDescriptor::operator=(const QueriableDescriptor& QD)
{
   _name = QD._name;
   _description = QD._description;
   _type = QD._type;
   _ptrQueriable = QD._ptrQueriable;
   return(*this);
}


std::string QueriableDescriptor::getName()
{
   return _name;
}


void QueriableDescriptor::setName(std::string name)
{
   _name = name;
}


std::string QueriableDescriptor::getDescription()
{
   return _description;
}


void QueriableDescriptor::setDescription (std::string description)
{
   _description = description;
}


std::string QueriableDescriptor::getType()
{
   return _type;
}


void QueriableDescriptor::setType(std::string type)
{
   _type = type;
}


Queriable* QueriableDescriptor::getQueriable()
{
   return _ptrQueriable;
}


void QueriableDescriptor::setQueriable(Queriable* ptrQueriable)
{
   _ptrQueriable = ptrQueriable;
}


QueriableDescriptor::~QueriableDescriptor()
{
}
