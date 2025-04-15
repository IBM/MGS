// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_query.h"
#include "C_query_field_entry.h"
#include "C_query_field_set.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_query::internalExecute(LensContext *c)
{
   if (_entry) _entry->execute(c);
   if (_set) _set->execute(c);
}

C_query::C_query(const C_query& rv)
   : C_production(rv), _entry(0), _set(0), _type(rv._type)
{
   if (rv._entry) {
      _entry = rv._entry->duplicate();
   }
   if (rv._set) {
      _set = rv._set->duplicate();
   }
}


C_query::C_query (C_query_field_entry* e, SyntaxError * error)
   : C_production(error), _entry(e), _set(0), _type(_ENTRY)
{
}


C_query::C_query (C_query_field_set* s, SyntaxError * error)
   : C_production(error), _entry(0), _set(s), _type(_SET)
{   
}


C_query* C_query::duplicate() const
{
   return new C_query(*this);
}


C_query::~C_query()
{
   delete _entry;
   delete _set;
}

void C_query::checkChildren() 
{
   if (_entry) {
      _entry->checkChildren();
      if (_entry->isError()) {
         setError();
      }
   }
   if (_set) {
      _set->checkChildren();
      if (_set->isError()) {
         setError();
      }
   }
} 

void C_query::recursivePrint() 
{
   if (_entry) {
      _entry->recursivePrint();
   }
   if (_set) {
      _set->recursivePrint();
   }
   printErrorMessage();
} 
