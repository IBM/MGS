// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_NDPAIR_CLAUSE_LIST_H
#define C_ARGUMENT_NDPAIR_CLAUSE_LIST_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_argument;
class C_ndpair_clause_list;
class LensContext;
class DataItem;
class NDPairListDataItem;
class SyntaxError;

class C_argument_ndpair_clause_list: public C_argument
{
   public:
      C_argument_ndpair_clause_list(const C_argument_ndpair_clause_list&);
      C_argument_ndpair_clause_list(C_ndpair_clause_list *, SyntaxError *);
      virtual ~C_argument_ndpair_clause_list();
      virtual C_argument_ndpair_clause_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_ndpair_clause_list *getNdpair_clause_list() { 
	 return _ndp_clause_list; 
      }
      DataItem *getArgumentDataItem() const;

   private:
      NDPairListDataItem* _ndpl_dataitem;
      C_ndpair_clause_list* _ndp_clause_list;
};
#endif
