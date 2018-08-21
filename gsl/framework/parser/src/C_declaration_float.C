// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_declaration_float.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_constant.h"
#include "FloatDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_float::internalExecute(LensContext *c)
{
   _constant->execute(c);
   _declarator->execute(c);
   _floatValue = _constant->getFloat();

   // now transfer _floatValue to a DataItem
   FloatDataItem *fdi =new FloatDataItem;
   fdi->setFloat(_floatValue);

   std::auto_ptr<DataItem> fdi_ap(static_cast<DataItem*>(fdi));
   try {
      c->symTable.addEntry(_declarator->getName(), fdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring float, " + e.getError());
   }
}


C_declaration_float* C_declaration_float::duplicate() const
{
   return new C_declaration_float(*this);
}


C_declaration_float::~C_declaration_float()
{
   delete _constant;
   delete _declarator;
}


C_declaration_float::C_declaration_float(const C_declaration_float& rv)
   : C_declaration(rv), _constant(0), _declarator(0), 
     _floatValue(rv._floatValue)
{
   if (rv._constant) {
      _constant = rv._constant;
   }
   if (rv._declarator) {
      _declarator = rv._declarator;
   }
}


C_declaration_float::C_declaration_float(
   C_declarator *d, C_constant *c, SyntaxError * error)
   : C_declaration(error), _constant(c), _declarator(d), _floatValue(0)
{
}

void C_declaration_float::checkChildren() 
{
   if (_constant) {
      _constant->checkChildren();
      if (_constant->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_declaration_float::recursivePrint() 
{
   if (_constant) {
      _constant->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
