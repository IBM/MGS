// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_NDPAIR_CLAUSE_H
#define C_ARGUMENT_NDPAIR_CLAUSE_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_argument;
class C_ndpair_clause;
class GslContext;
class DataItem;
class NDPairDataItem;
class SyntaxError;

class C_argument_ndpair_clause: public C_argument
{
   public:
      C_argument_ndpair_clause(const C_argument_ndpair_clause&);
      C_argument_ndpair_clause(C_ndpair_clause *, SyntaxError *);
      virtual ~C_argument_ndpair_clause();
      virtual C_argument_ndpair_clause* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_ndpair_clause *getNdpair_clause() { 
	 return _ndp_clause; 
      }
      DataItem *getArgumentDataItem() const;

   private:
      NDPairDataItem* _ndp_dataitem;
      C_ndpair_clause* _ndp_clause;
};
#endif
