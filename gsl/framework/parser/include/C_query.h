// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_query_H
#define C_query_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class GslContext;
class C_query_field_entry;
class C_query_field_set;
class SyntaxError;

class C_query : public C_production
{
   public:
      enum Type {_ENTRY, _SET};
      C_query(C_query const &);
      C_query(C_query_field_entry *, SyntaxError *);
      C_query(C_query_field_set *, SyntaxError *);
      virtual ~C_query ();
      virtual C_query* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_query_field_entry* getEntry() {
	 return _entry;
      }
      C_query_field_set* getSet() {
	 return _set;
      }
      Type getType() {
	 return _type;
      }

   private:
      C_query_field_entry* _entry;
      C_query_field_set* _set;
      Type _type;
};
#endif
