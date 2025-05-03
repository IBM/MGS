// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_list.h"
#include "C_type_specifier.h"
#include "C_argument_argument_list.h"
#include "DataItem.h"
#include "FloatDataItem.h"
#include "IntDataItem.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "DataItemArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"
#include "ArgumentListHelper.h"

void C_argument_argument_list::internalExecute(GslContext *c)
{
   _arg_list->execute(c);
   if (_typeSpec) {
      _typeSpec->execute(c);
   }
   std::unique_ptr<DataItem> diap;
   ArgumentListHelper helper;
   helper.getDataItem(diap, c, _arg_list, _typeSpec);
   _dataItem = diap.release();
}

C_argument_argument_list::C_argument_argument_list(
   const C_argument_argument_list& rv)
   : C_argument(rv), _dataItem(0), _arg_list(0), _typeSpec(rv._typeSpec)
{
   if (rv._arg_list) {
      _arg_list = rv._arg_list->duplicate();
   }
   if (rv._dataItem) {
      std::unique_ptr<DataItem> di;
      rv._dataItem->duplicate(di);
      _dataItem = di.release();
   }
}

C_argument_argument_list::C_argument_argument_list(
   C_argument_list *al, SyntaxError *error)
   : C_argument(_ARG_LIST, error), _dataItem(0), _arg_list(al), _typeSpec(0)
{
}

C_argument_argument_list::C_argument_argument_list(
   C_type_specifier *t, C_argument_list *al, SyntaxError *error)
   : C_argument(_ARG_LIST, error), _dataItem(0), _arg_list(al), _typeSpec(t)
{
}

C_argument_argument_list* C_argument_argument_list::duplicate() const
{
   return new C_argument_argument_list(*this);
}

C_argument_argument_list::~C_argument_argument_list()
{
   delete _arg_list;
   delete _dataItem;
   delete _typeSpec;
}

void C_argument_argument_list::checkChildren() 
{
   if (_arg_list) {
      _arg_list->checkChildren();
      if (_arg_list->isError()) {
         setError();
      }
   }
   if (_typeSpec) {
      _typeSpec->checkChildren();
      if (_typeSpec->isError()) {
         setError();
      }
   }
} 

void C_argument_argument_list::recursivePrint() 
{
   if (_arg_list) {
      _arg_list->recursivePrint();
   }
   if (_typeSpec) {
      _typeSpec->recursivePrint();
   }
   printErrorMessage();
} 
