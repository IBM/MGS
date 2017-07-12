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

#ifndef C_query_field_entry_list_H
#define C_query_field_entry_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_query_field_entry;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
