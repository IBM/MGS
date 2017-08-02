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

#include "C_matrix_type_specifier.h"
#include "C_type_specifier.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_matrix_type_specifier::internalExecute(LensContext *c)
{
   _typeSpecifier->execute(c);
}


C_matrix_type_specifier::C_matrix_type_specifier(
   const C_matrix_type_specifier& rv)
   : C_production(rv), _typeSpecifier(0)
{
   if (rv._typeSpecifier) {
      _typeSpecifier = rv._typeSpecifier->duplicate();
   }
}


C_matrix_type_specifier::C_matrix_type_specifier(
   C_type_specifier *ts, SyntaxError * error)
   : C_production(error), _typeSpecifier(ts)
{
}


C_matrix_type_specifier* C_matrix_type_specifier::duplicate() const
{
   return new C_matrix_type_specifier(*this);
}


C_matrix_type_specifier::~C_matrix_type_specifier()
{
   delete _typeSpecifier;
}


C_type_specifier * C_matrix_type_specifier::getTypeSpecifier() const
{
   return _typeSpecifier;
}

void C_matrix_type_specifier::checkChildren() 
{
   if (_typeSpecifier) {
      _typeSpecifier->checkChildren();
      if (_typeSpecifier->isError()) {
         setError();
      }
   }
} 

void C_matrix_type_specifier::recursivePrint() 
{
   if (_typeSpecifier) {
      _typeSpecifier->recursivePrint();
   }
   printErrorMessage();
} 
