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

#include "C_query_field_entry.h"
#include "C_ndpair_clause.h"
#include "NDPair.h"
#include "C_constant.h"
#include "C_declarator.h"
#include <sstream>
#include "SyntaxError.h"
#include "C_production.h"

void C_query_field_entry::internalExecute(LensContext *c)
{
   if (_ndpClause) _ndpClause->execute(c);
   if (_constant) _constant->execute(c);
   if (_declarator) _declarator->execute(c);

   if (_ndpClause) {
      _fieldName = _ndpClause->getNDPair().getName();
      _entry = new std::string(_ndpClause->getNDPair().getValue());
   }
   else if (_constant) {
      std::ostringstream i;
      if (_constant->getType() == C_constant::_INT) i<<_constant->getInt();
      if (_constant->getType() == C_constant::_FLOAT) i<<_constant->getFloat();
      _entry = new std::string(i.str());
   }
   else if (_declarator) {
      _entry = new std::string (_declarator->getName());
   }
}

C_query_field_entry::C_query_field_entry(const C_query_field_entry& rv)
: C_production(rv), _fieldName(rv._fieldName), _entry(0), _ndpClause(0), _constant(0), 
  _declarator(0)
{
   if (rv._ndpClause) {
      _ndpClause = rv._ndpClause->duplicate();
   }
   if (rv._constant) {
      _constant = rv._constant->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._entry){
      _entry = new std::string(*(rv._entry));
   }
}


C_query_field_entry::C_query_field_entry(C_ndpair_clause* ndpc, 
					 SyntaxError * error)
   : C_production(error), _fieldName(""), _entry(0), _ndpClause(ndpc), 
     _constant(0), _declarator(0)
{
}


C_query_field_entry::C_query_field_entry(std::string *entry, 
					 SyntaxError * error)
   : C_production(error), _fieldName(""), _entry(entry), _ndpClause(0), 
     _constant(0), _declarator(0)
{
}


C_query_field_entry::C_query_field_entry(C_constant* c, SyntaxError * error)
   : C_production(error), _fieldName(""), _entry(0), _ndpClause(0), 
     _constant(c), _declarator(0)
{
}


C_query_field_entry::C_query_field_entry(C_declarator* d, SyntaxError * error)
   : C_production(error), _fieldName(""), _entry(0), _ndpClause(0),
     _constant(0), _declarator(d)
{
}


C_query_field_entry* C_query_field_entry::duplicate() const
{
   return new C_query_field_entry(*this);
}


C_query_field_entry::~C_query_field_entry()
{
   delete _ndpClause;
   delete _constant;
   delete _declarator;
   delete _entry;
}

void C_query_field_entry::checkChildren() 
{
   if (_ndpClause) {
      _ndpClause->checkChildren();
      if (_ndpClause->isError()) {
         setError();
      }
   }
   if (_constant) {
      _constant->checkChildren();
      if (_constant->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_query_field_entry::recursivePrint() 
{
   if (_ndpClause) {
      _ndpClause->recursivePrint();
   }
   if (_constant) {
      _constant->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
