// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_ndpair_clause.h"
#include "C_ndpair_clause.h"
#include "DataItem.h"
#include "NDPairDataItem.h"
#include "NDPair.h"
#include "SyntaxError.h"

void C_argument_ndpair_clause::internalExecute(GslContext *c)
{
   _ndp_clause->execute(c);

   std::unique_ptr<NDPair> ndp;
   _ndp_clause->releaseNDPair(ndp);
   delete _ndp_dataitem;
   _ndp_dataitem = new NDPairDataItem;
   _ndp_dataitem->setNDPair(ndp);
}


C_argument_ndpair_clause::C_argument_ndpair_clause(
   const C_argument_ndpair_clause& rv)
   : C_argument(rv), _ndp_dataitem(0), _ndp_clause(0)
{
   if (rv._ndp_clause) {
      _ndp_clause = rv._ndp_clause->duplicate();
   }
   if (rv._ndp_dataitem) {
      std::unique_ptr<DataItem> cc_di;
      rv._ndp_dataitem->duplicate(cc_di);
      _ndp_dataitem = dynamic_cast<NDPairDataItem*>(cc_di.release());
   }
}


C_argument_ndpair_clause::C_argument_ndpair_clause(
   C_ndpair_clause *n, SyntaxError * error)
   : C_argument(_NDPAIR, error), _ndp_dataitem(0), _ndp_clause(n)
{
}


C_argument_ndpair_clause* C_argument_ndpair_clause::duplicate() const
{
   return new C_argument_ndpair_clause(*this);
}


C_argument_ndpair_clause::~C_argument_ndpair_clause()
{
   delete _ndp_clause;
   delete _ndp_dataitem;
}


DataItem* C_argument_ndpair_clause::getArgumentDataItem() const
{
   return _ndp_dataitem;
}

void C_argument_ndpair_clause::checkChildren() 
{
   if (_ndp_clause) {
      _ndp_clause->checkChildren();
      if (_ndp_clause->isError()) {
         setError();
      }
   }
} 

void C_argument_ndpair_clause::recursivePrint() 
{
   if (_ndp_clause) {
      _ndp_clause->recursivePrint();
   }
   printErrorMessage();
} 
