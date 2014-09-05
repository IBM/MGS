// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_argument_ndpair_clause.h"
#include "C_ndpair_clause.h"
#include "DataItem.h"
#include "NDPairDataItem.h"
#include "NDPair.h"
#include "SyntaxError.h"

void C_argument_ndpair_clause::internalExecute(LensContext *c)
{
   _ndp_clause->execute(c);

   std::auto_ptr<NDPair> ndp;
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
      std::auto_ptr<DataItem> cc_di;
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
