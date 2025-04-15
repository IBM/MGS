// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_complex_functor_declaration_body.h"
#include "C_function_clause.h"
#include "C_constructor_clause.h"
#include "C_return_clause.h"
#include "C_parameter_type.h"
#include "C_complex_functor_clause_list.h"
#include "C_complex_functor_clause.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_complex_functor_declaration_body::internalExecute(LensContext *c)
{
   if(_constructorClause) {
      _constructorClause->execute(c);
      _constructor_ptl = _constructorClause->getParameterTypeList();
   }
   else {
      _constructor_ptl = &_empty;
   }
   if(_functionClause) {
      _functionClause->execute(c);
      _function_ptl = _functionClause->getParameterTypeList();
   }
   else {
      _function_ptl = &_empty;
   }
   if(_returnClause) {
      _returnClause->execute(c);
      _return_ptl = _returnClause->getParameterTypeList();
   }
   else {
      _return_ptl = &_empty;
   }
}


C_complex_functor_declaration_body::C_complex_functor_declaration_body(
   const C_complex_functor_declaration_body& rv)
   : C_production(rv), _constructor_ptl(0), _function_ptl(0), _return_ptl(0), 
     _empty(rv._empty), _constructorClause(0), _functionClause(0), 
     _returnClause(0), _complexFunctorClauseList(0), 
     _ownPointers(rv._ownPointers)
{
   if (rv._constructorClause) {
      _constructorClause = rv._constructorClause->duplicate();
   }
   if (rv._functionClause) {
      _functionClause = rv._functionClause->duplicate();
   }
   if (rv._returnClause) {
      _returnClause = rv._returnClause->duplicate();
   }
   if (rv._complexFunctorClauseList) {
      _complexFunctorClauseList = rv._complexFunctorClauseList->duplicate();
   }
}


C_complex_functor_declaration_body::C_complex_functor_declaration_body (
   C_complex_functor_clause_list *l, SyntaxError * error)
   : C_production(error), _constructor_ptl(0), _function_ptl(0), 
     _return_ptl(0), _constructorClause(0), _functionClause(0), 
     _returnClause(0), _complexFunctorClauseList(0), _ownPointers(false)
{
   // KLUDGE: keeps only last parameter type list of a particular type
   // fix: beef up a cascade of interfaces to supports lists of 
   // std::list<parameter_type>.  ie C_definition_functor, 
   // C_functor_definition, ScriptFunctorTypeDataItem...

   _complexFunctorClauseList = l;
   C_complex_functor_clause* current=0;
   std::list<C_complex_functor_clause*>::iterator i, end = l->getList()->end();
   for (i = l->getList()->begin(); i != end; ++i) {
      current = *i;
      switch(current->getType()) {
         case (C_complex_functor_clause::_CONSTRUCTOR):
            _constructorClause = dynamic_cast<C_constructor_clause*>(current);
            break;
         case (C_complex_functor_clause::_FUNCTION):
            _functionClause = dynamic_cast<C_function_clause*>(current);
            break;
         case (C_complex_functor_clause::_RETURN):
            _returnClause = dynamic_cast<C_return_clause*>(current);
            break;
      }
   }
}


C_complex_functor_declaration_body::C_complex_functor_declaration_body (
   C_constructor_clause *c, C_function_clause *f, C_return_clause *r, 
   SyntaxError * error)
   : C_production(error), _constructor_ptl(0), _function_ptl(0), 
     _return_ptl(0), _constructorClause(c), _functionClause(f), 
     _returnClause(r), _complexFunctorClauseList(0), _ownPointers(true)
{
}


C_complex_functor_declaration_body* 
C_complex_functor_declaration_body::duplicate() const
{
   return new C_complex_functor_declaration_body(*this);
}


C_complex_functor_declaration_body::~C_complex_functor_declaration_body()
{
   if (_ownPointers) {
      delete _constructorClause;
      delete _functionClause;
      delete _returnClause;
   }
   delete _complexFunctorClauseList;
}

void C_complex_functor_declaration_body::checkChildren() 
{
   if (_constructorClause) {
      _constructorClause->checkChildren();
      if (_constructorClause->isError()) {
         setError();
      }
   }
   if (_functionClause) {
      _functionClause->checkChildren();
      if (_functionClause->isError()) {
         setError();
      }
   }
   if (_returnClause) {
      _returnClause->checkChildren();
      if (_returnClause->isError()) {
         setError();
      }
   }
   if (_complexFunctorClauseList) {
      _complexFunctorClauseList->checkChildren();
      if (_complexFunctorClauseList->isError()) {
         setError();
      }
   }
} 

void C_complex_functor_declaration_body::recursivePrint() 
{
   if (_constructorClause) {
      _constructorClause->recursivePrint();
   }
   if (_functionClause) {
      _functionClause->recursivePrint();
   }
   if (_returnClause) {
      _returnClause->recursivePrint();
   }
   if (_complexFunctorClauseList) {
      _complexFunctorClauseList->recursivePrint();
   }
   printErrorMessage();
} 
