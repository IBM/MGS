// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_query_field_set_H
#define C_query_field_set_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class C_query_field_entry_list;
class SyntaxError;

class C_query_field_set : public C_production
{
   public:
      C_query_field_set(const C_query_field_set&);
      C_query_field_set(C_query_field_entry_list *, SyntaxError *);
      virtual ~C_query_field_set ();
      virtual C_query_field_set* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_query_field_entry_list* getQueryFieldEntryList() {
	 return _queryFieldEntryList;
      }

   private:
      C_query_field_entry_list* _queryFieldEntryList;
};
#endif
