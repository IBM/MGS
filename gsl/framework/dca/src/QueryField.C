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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "QueryField.h"
#include "EnumEntry.h"

#include <iostream>

QueryField::QueryField(queryType type)
{
   _type = type;
   _field = "";
   _name = "";
   _description = "";
   _format = "";
}


QueryField::QueryField(QueryField* qf)
: _type(qf->_type), _field(qf->_field), _name(qf->_name), _description(qf->_description),
_format(qf->_format)
{
   if (_type == ENUM) {
      std::vector<EnumEntry*>::iterator end = qf->_enumEntries.end();
      for (std::vector<EnumEntry*>::iterator iter = qf->_enumEntries.begin(); iter != end; iter++) {
         _enumEntries.push_back(new EnumEntry(*iter));
      }
   }
}


std::string QueryField::getField()
{
   return _field;
}


void QueryField::setField(std::string field)
{
   _field = field;
}


std::string QueryField::getName()
{
   return _name;
}


void QueryField::setName(std::string name)
{
   _name = name;
}


std::string QueryField::getDescription()
{
   return _description;
}


void QueryField::setDescription(std::string description)
{
   _description = description;
}


QueryField::queryType QueryField::getType()
{
   return _type;
}


std::vector<EnumEntry*> const & QueryField::getEnumEntries() const
{

   if (_type == VALUE) std::cerr<<"Enum Entrier requested from VALUE type Query Field!"<<std::endl;
   return _enumEntries;
}


void QueryField::addEnumEntry(std::auto_ptr<EnumEntry> & aptrEnumEntry)
{
   if (_type == ENUM) _enumEntries.push_back(aptrEnumEntry.release());
   if (_type == VALUE) std::cerr<<"Cannot add enum entry to VALUE type Query Field!"<<std::endl;
}


std::string QueryField::getFormat()
{
   return _format;
}


void QueryField::setFormat(std::string format)
{
   _format = format;
}


bool QueryField::isSet()
{
   return _field != "";
}


QueryField::~QueryField()
{
   if (_type == ENUM) {
      std::vector<EnumEntry*>::iterator end = _enumEntries.end();
      for (std::vector<EnumEntry*>::iterator iter = _enumEntries.begin(); iter != end; iter++) {
         delete (*iter);
      }
   }
}
