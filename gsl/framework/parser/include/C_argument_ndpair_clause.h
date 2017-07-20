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

#ifndef C_ARGUMENT_NDPAIR_CLAUSE_H
#define C_ARGUMENT_NDPAIR_CLAUSE_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_argument;
class C_ndpair_clause;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
