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
