// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_edgeset_extension.h"
#include "C_declarator.h"
#include "C_index_set_specifier.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_edgeset_extension::internalExecute(LensContext *c)
{
   if(_indexSetSpecifier) {
      _indexSetSpecifier->execute(c);
      _indices = _indexSetSpecifier->getIndices();
   }
   if (_declarator) {
      _declarator->execute(c);
      _type = _declarator->getName();
   }
}


C_edgeset_extension::C_edgeset_extension(const C_edgeset_extension& rv)
   : C_production(rv), _declarator(0), _indexSetSpecifier(0), _type(rv._type),
     _indices(rv._indices)

{
   if(rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if(rv._indexSetSpecifier) {
      _indexSetSpecifier = rv._indexSetSpecifier->duplicate();
   }
}


C_edgeset_extension::C_edgeset_extension(C_declarator *decl, SyntaxError * error)
   : C_production(error), _declarator(decl), _indexSetSpecifier(0)
{
}


C_edgeset_extension::C_edgeset_extension(
   C_declarator *decl, C_index_set_specifier *i, SyntaxError * error)
   : C_production(error), _declarator(decl), _indexSetSpecifier(i)
{
}


C_edgeset_extension::C_edgeset_extension(C_index_set_specifier *i, SyntaxError * error)
   : C_production(error), _declarator(0), _indexSetSpecifier(i)
{
}


C_edgeset_extension* C_edgeset_extension::duplicate() const
{
   return new C_edgeset_extension(*this);
}


C_edgeset_extension::~C_edgeset_extension()
{
   delete _declarator;
   delete _indexSetSpecifier;
}

void C_edgeset_extension::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_indexSetSpecifier) {
      _indexSetSpecifier->checkChildren();
      if (_indexSetSpecifier->isError()) {
         setError();
      }
   }
} 

void C_edgeset_extension::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_indexSetSpecifier) {
      _indexSetSpecifier->recursivePrint();
   }
   printErrorMessage();
} 
