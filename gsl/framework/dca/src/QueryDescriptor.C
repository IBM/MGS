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

#include "QueryDescriptor.h"
#include "QueryField.h"

QueryDescriptor::QueryDescriptor()
{
}


QueryDescriptor::QueryDescriptor(const QueryDescriptor& qd)
//  : _anyFieldSet(qd._anyFieldSet)
// Not set, so ZF was complaining.
{
   std::vector<QueryField*>::const_iterator it, end = qd._fields.end();
   for (it = qd._fields.begin(); it != end; it++) {
      _fields.push_back(new QueryField(*it));
   }
}


QueryDescriptor& QueryDescriptor::operator=(const QueryDescriptor& QD)
{
   _anyFieldSet = QD._anyFieldSet;
   // !!!!!!! Remember to delete the contents, PTNTL MEM LEAK
   for (std::vector<QueryField*>::const_iterator i = QD._fields.begin();
   i != QD._fields.end(); i++) {
      _fields.insert(_fields.end(), new QueryField(*i));
      // !!!!! Make a duplicate method for QueryField

   }
   return(*this);
}


void QueryDescriptor::clearFields()
{
   std::vector<QueryField*>::iterator end = _fields.end();
   for (std::vector<QueryField*>::iterator iter = _fields.begin(); iter != end; iter++) {
      (*iter)->setField("");
   }
}


int QueryDescriptor::addQueryField(std::auto_ptr<QueryField> & qfield)
{
   _fields.push_back(qfield.release());
   return (_fields.size())-1;
}


std::vector<QueryField*> & QueryDescriptor::getQueryFields()
{
   return _fields;
}


bool QueryDescriptor::isAnyFieldSet()
{
   std::vector<QueryField*>::iterator end = _fields.end();
   for (std::vector<QueryField*>::iterator iter = _fields.begin(); iter != end; iter++) {
      if ((*iter)->getField() != "") return true;
   }
   return false;
}


QueryDescriptor::~QueryDescriptor()
{
   std::vector<QueryField*>::iterator end = _fields.end();
   for (std::vector<QueryField*>::iterator iter = _fields.begin(); iter != end; iter++) {
      delete (*iter);
   }
}
