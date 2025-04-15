// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_ndpair_clause_list.h"
#include "C_ndpair_clause_list.h"
#include "DataItem.h"
#include "NDPairListDataItem.h"
#include "NDPairList.h"
#include "SyntaxError.h"
#include <cassert>

void C_argument_ndpair_clause_list::internalExecute(LensContext *c)
{
   if (_ndp_clause_list) {
      _ndp_clause_list->execute(c);

      std::unique_ptr<NDPairList> ndp;
      _ndp_clause_list->releaseList(ndp);   
      _ndpl_dataitem = new NDPairListDataItem;
      _ndpl_dataitem->setNDPairList(ndp);
   } else {
      assert(_ndpl_dataitem != 0);
   }
}


C_argument_ndpair_clause_list::C_argument_ndpair_clause_list(
   const C_argument_ndpair_clause_list& rv)
   : C_argument(rv), _ndpl_dataitem(0), _ndp_clause_list(0)
{
   if (rv._ndp_clause_list) {
      _ndp_clause_list = rv._ndp_clause_list->duplicate();
   }
   if (rv._ndpl_dataitem) {
      std::unique_ptr<DataItem> cc_di;
      rv._ndpl_dataitem->duplicate(cc_di);
      _ndpl_dataitem = dynamic_cast<NDPairListDataItem*>(cc_di.release());
   }
}


C_argument_ndpair_clause_list::C_argument_ndpair_clause_list(
   C_ndpair_clause_list *n,SyntaxError *error)
   : C_argument(_NDPAIRLIST, error), _ndpl_dataitem(0), _ndp_clause_list(n)
{
}


C_argument_ndpair_clause_list* C_argument_ndpair_clause_list::duplicate() const
{
   return new C_argument_ndpair_clause_list(*this);
}


C_argument_ndpair_clause_list::~C_argument_ndpair_clause_list()
{
   delete _ndp_clause_list;
   delete _ndpl_dataitem;
}


DataItem* C_argument_ndpair_clause_list::getArgumentDataItem() const
{
   return _ndpl_dataitem;
}

void C_argument_ndpair_clause_list::checkChildren() 
{
   if (_ndp_clause_list) {
      _ndp_clause_list->checkChildren();
      if (_ndp_clause_list->isError()) {
         setError();
      }
   }
} 

void C_argument_ndpair_clause_list::recursivePrint() 
{
   if (_ndp_clause_list) {
      _ndp_clause_list->recursivePrint();
   }
   printErrorMessage();
} 
