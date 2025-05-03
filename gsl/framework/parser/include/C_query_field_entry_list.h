// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_query_field_entry_list_H
#define C_query_field_entry_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_query_field_entry;
class GslContext;
class SyntaxError;

class C_query_field_entry_list : public C_production
{
   public:
      C_query_field_entry_list(const C_query_field_entry_list&);
      C_query_field_entry_list(C_query_field_entry *, SyntaxError *);
      C_query_field_entry_list(C_query_field_entry_list *, 
			       C_query_field_entry *, 
			       SyntaxError *);
      virtual C_query_field_entry_list* duplicate() const;
      std::list<C_query_field_entry>* releaseList();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_query_field_entry_list ();
      std::list<C_query_field_entry>* getList() const {
	 return _list;
      }

   private:
      std::list<C_query_field_entry>* _list;
};
#endif
