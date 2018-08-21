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

#include "C_non_initializable_type_specifier.h"
#include "C_type_specifier.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_non_initializable_type_specifier::internalExecute(LensContext *c)
{
}


C_non_initializable_type_specifier::C_non_initializable_type_specifier(
   const C_non_initializable_type_specifier& rv)
   : C_production(rv), _type(rv._type), _nextTypeSpec(0)
{
}


C_non_initializable_type_specifier::C_non_initializable_type_specifier(
   C_type_specifier::Type t, SyntaxError * error)
   : C_production(error), _type(t), _nextTypeSpec(0)
{
}


C_non_initializable_type_specifier* 
C_non_initializable_type_specifier::duplicate() const
{
   return new C_non_initializable_type_specifier(*this);
}


C_non_initializable_type_specifier::~C_non_initializable_type_specifier()
{
}

void C_non_initializable_type_specifier::checkChildren() 
{
   if (_nextTypeSpec) {
      _nextTypeSpec->checkChildren();
      if (_nextTypeSpec->isError()) {
         setError();
      }
   }
} 

void C_non_initializable_type_specifier::recursivePrint() 
{
   if (_nextTypeSpec) {
      _nextTypeSpec->recursivePrint();
   }
   printErrorMessage();
} 
