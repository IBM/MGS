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
