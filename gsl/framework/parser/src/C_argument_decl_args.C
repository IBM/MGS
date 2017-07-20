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

#include "C_argument_decl_args.h"
#include "LensContext.h"
#include "C_argument_list.h"
#include "C_declarator.h"
#include "Functor.h"
#include "FunctorType.h"
#include "InstanceFactory.h"
#include "FunctorDataItem.h"
#include "FunctorTypeDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_argument_decl_args::internalExecute(LensContext *c)
{

   _declarator->execute(c);
   _argList->execute(c);

   // Get entry from symbol table
   std::string name = _declarator->getName();
   const DataItem *symTableDI = c->symTable.getEntry(name);

   if( !symTableDI ) {
      std::string mes = name + " has not been declared";
      throwError(mes);
   }
   if (symTableDI->getType()==FunctorTypeDataItem::_type) {
      const FunctorTypeDataItem *ftdi = 
	 dynamic_cast<const FunctorTypeDataItem*>(symTableDI);
      if (ftdi ==0) {
	 std::string mes = "Unable to cast Functor type " + 
	    _declarator->getName();
	 throwError(mes);
      }
      InstanceFactory* ifc = ftdi->getInstanceFactory();
      std::auto_ptr<DataItem> apdi;
      ifc->getInstance(apdi, _argList->getVectorDataItem(),c);
      DataItem* di = apdi.release();
      delete _dataItem;
      _dataItem = dynamic_cast<FunctorDataItem*>(di);
      if (_dataItem == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to FunctorDataItem failed";
	 throwError(mes);
      }
   }
   else if (symTableDI->getType()==FunctorDataItem::_type) {
      const FunctorDataItem *fdi = 
	 dynamic_cast<const FunctorDataItem*>(symTableDI);
      if (fdi == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to FunctorDataItem failed";
	 throwError(mes);
      }
      Functor* f = fdi->getFunctor();
      std::auto_ptr<DataItem> rval;
      f->execute(c,*(_argList->getVectorDataItem()), rval);
      delete _dataItem;
      _dataItem = rval.release();

   }
   else {
      std::string mes = " " + name + 
	 " was previously declared not as a functor or functor type.";
      throwError(mes);
   }
}

C_argument_decl_args::C_argument_decl_args(const C_argument_decl_args& rv)
   : C_argument(rv), _declarator(0), _argList(0), _dataItem(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._argList) {
      _argList = rv._argList->duplicate();
   }
   if (rv._dataItem) {
      std::auto_ptr<DataItem> cc_di;
      rv._dataItem->duplicate(cc_di);
      _dataItem = cc_di.release();
   }
}

C_argument_decl_args::C_argument_decl_args(
   C_declarator *d, C_argument_list *a, SyntaxError * error)
   : C_argument(_DECL_ARGS, error), _declarator(d), _argList(a),_dataItem(0)
{
}

C_argument_decl_args* C_argument_decl_args::duplicate() const
{
   return new C_argument_decl_args(*this);
}

C_argument_decl_args::~C_argument_decl_args()
{
   delete _declarator;
   delete _argList;
   delete _dataItem;
}

void C_argument_decl_args::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_argList) {
      _argList->checkChildren();
      if (_argList->isError()) {
         setError();
      }
   }
} 

void C_argument_decl_args::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_argList) {
      _argList->recursivePrint();
   }
   printErrorMessage();
} 
