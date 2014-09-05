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

#include "C_initializable_type_specifier.h"
#include "LensContext.h"
#include "C_parameter_type_pair.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_initializable_type_specifier::internalExecute(LensContext *c)
{
   if (_parameterTypePair)
      _parameterTypePair->execute(c);

   if (_typeSpec)
      _typeSpec->execute(c);

}


C_initializable_type_specifier::C_initializable_type_specifier(
   const C_initializable_type_specifier& rv)
   : C_production(rv), _type(rv._type), _parameterTypePair(0), _typeSpec(0)
{
   if (rv._parameterTypePair) {
      _parameterTypePair = rv._parameterTypePair->duplicate();
   }
   if (rv._typeSpec) {
      _typeSpec = rv._typeSpec->duplicate();
   }
}


C_initializable_type_specifier::C_initializable_type_specifier(
   C_type_specifier::Type t, SyntaxError * error)
   : C_production(error), _type(t), _parameterTypePair(0), _typeSpec(0)
{
}


C_initializable_type_specifier::C_initializable_type_specifier(
   C_type_specifier::Type t, C_parameter_type_pair *p, SyntaxError * error)
   : C_production(error), _type(t), _parameterTypePair(p), _typeSpec(0)
{
}


C_initializable_type_specifier::C_initializable_type_specifier(
   C_type_specifier::Type t, C_type_specifier *ts, SyntaxError * error)
   : C_production(error), _type(t), _parameterTypePair(0), _typeSpec(ts)
{
}


C_initializable_type_specifier* 
C_initializable_type_specifier::duplicate() const
{
   return new C_initializable_type_specifier(*this);
}

C_initializable_type_specifier::~C_initializable_type_specifier()
{
   delete _parameterTypePair;
   delete _typeSpec;
}

void C_initializable_type_specifier::checkChildren() 
{
   if (_parameterTypePair) {
      _parameterTypePair->checkChildren();
      if (_parameterTypePair->isError()) {
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

void C_initializable_type_specifier::recursivePrint() 
{
   if (_parameterTypePair) {
      _parameterTypePair->recursivePrint();
   }
   if (_typeSpec) {
      _typeSpec->recursivePrint();
   }
   printErrorMessage();
} 
