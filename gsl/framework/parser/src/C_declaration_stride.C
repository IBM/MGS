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

#include "C_declaration_stride.h"
#include "C_declarator.h"
#include "C_stride_list.h"
#include "LensContext.h"
#include "StridesListDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_stride::internalExecute(LensContext *c)
{
   _cDeclarator->execute(c);
   _cStrideList->execute(c);

   _stridesList = _cStrideList->getStridesList();

   // Now transfer StridesList to a DataItem
   StridesListDataItem *sldi = new StridesListDataItem;
   // sldi->setStridesList(_stridesList);
   sldi->setStridesList(_cStrideList);

   std::auto_ptr<DataItem> sldi_ap(static_cast<DataItem*>(sldi));
   try {
      c->symTable.addEntry(_cDeclarator->getName(), sldi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring stride, " + e.getError());
   }
}


C_declaration_stride* C_declaration_stride::duplicate() const
{
   return new C_declaration_stride(*this);
}


C_declaration_stride::C_declaration_stride(
   C_declarator *d, C_stride_list *s, SyntaxError * error)
   : C_declaration(error), _cDeclarator(d), _cStrideList(s)
{
}


C_declaration_stride::C_declaration_stride(const C_declaration_stride& rv)
   : C_declaration(rv), _cDeclarator(0), _cStrideList(0)
{
   if (rv._cDeclarator) {
      _cDeclarator = rv._cDeclarator->duplicate();
   }
   if (rv._cStrideList) {
      _cStrideList = rv._cStrideList->duplicate();
   }
}


C_declaration_stride::~C_declaration_stride()
{
   delete _cDeclarator;
   delete _cStrideList;
}

void C_declaration_stride::checkChildren() 
{
   if (_cDeclarator) {
      _cDeclarator->checkChildren();
      if (_cDeclarator->isError()) {
         setError();
      }
   }
   if (_cStrideList) {
      _cStrideList->checkChildren();
      if (_cStrideList->isError()) {
         setError();
      }
   }
} 

void C_declaration_stride::recursivePrint() 
{
   if (_cDeclarator) {
      _cDeclarator->recursivePrint();
   }
   if (_cStrideList) {
      _cStrideList->recursivePrint();
   }
   printErrorMessage();
} 
