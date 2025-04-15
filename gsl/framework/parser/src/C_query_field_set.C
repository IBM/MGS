// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_query_field_set.h"
#include "C_query_field_entry_list.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_query_field_set::internalExecute(LensContext *c)
{
   _queryFieldEntryList->execute(c);
}


C_query_field_set::C_query_field_set(const C_query_field_set& rv)
   : C_production(rv)
{
   if (rv._queryFieldEntryList) {
      _queryFieldEntryList = rv._queryFieldEntryList->duplicate();
   }
}


C_query_field_set::C_query_field_set(C_query_field_entry_list* qfel, 
				     SyntaxError * error)
   : C_production(error), _queryFieldEntryList(qfel)
{
}


C_query_field_set* C_query_field_set::duplicate() const
{
   return new C_query_field_set(*this);
}


C_query_field_set::~C_query_field_set()
{
   delete _queryFieldEntryList;
}

void C_query_field_set::checkChildren() 
{
   if (_queryFieldEntryList) {
      _queryFieldEntryList->checkChildren();
      if (_queryFieldEntryList->isError()) {
         setError();
      }
   }
} 

void C_query_field_set::recursivePrint() 
{
   if (_queryFieldEntryList) {
      _queryFieldEntryList->recursivePrint();
   }
   printErrorMessage();
} 
