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

#include "C_functor_definition.h"
#include "LensContext.h"
#include "C_functor_category.h"
#include "C_parameter_type.h"
#include "C_parameter_type_list.h"
#include "C_declarator.h"
#include "C_complex_functor_definition.h"
#include "C_connection_script_definition.h"
#include "ConnectorFunctor.h"
#include "Simulation.h"
#include "ScriptFunctorType.h"
#include "FunctorType.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

#include <memory>
#include <iostream>

void C_functor_definition::internalExecute(LensContext *c)
{
   // call executes
   if(_functor_category) _functor_category->execute(c);
   if(_declarator) _declarator->execute(c);
   if(_constructor_ptl) _constructor_ptl->execute(c);
   if(_complex_functor_def) _complex_functor_def->execute(c);
   if(_c_script_def) _c_script_def->execute(c);

   // do work
   switch(_basis) {
      case _BASIC: basicWork(c); break;
      case _CONSTR_DEF: constrDefWork(c); break;
      case _COMPLEX: complexWork(c); break;
      case _SCRIPT: scriptWork(c); break;
   };
}


C_functor_definition::C_functor_definition(const C_functor_definition& rv)
   : C_production(rv), _basis(rv._basis), _functor_category(0), _declarator(0),
     _constructor_ptl(0), _complex_functor_def(0), _c_script_def(0), 
     _functorType(rv._functorType), _sft(0)
{
   if (rv._functor_category) {
      _functor_category = rv._functor_category->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._constructor_ptl) {
      _constructor_ptl = rv._constructor_ptl->duplicate();
   }
   if (rv._complex_functor_def) {
      _complex_functor_def = rv._complex_functor_def->duplicate();
   }
   if (rv._c_script_def) {
      _c_script_def = rv._c_script_def->duplicate();
   }
   if (rv._sft) {
      _sft = new ScriptFunctorType(rv._sft);
   }
}


C_functor_definition::C_functor_definition(
   C_functor_category *f, C_declarator *d, SyntaxError * error)
   : C_production(error), _basis(_BASIC), _functor_category(f), _declarator(d),
     _constructor_ptl(0), _complex_functor_def(0), _c_script_def(0), 
     _functorType(0), _sft(0)
{
}


C_functor_definition::C_functor_definition(
   C_functor_category *f, C_declarator *d, C_parameter_type_list *p, 
   SyntaxError * error)
   : C_production(error), _basis(_CONSTR_DEF), _functor_category(f), 
     _declarator(d), _constructor_ptl(p), _complex_functor_def(0), 
     _c_script_def(0), _functorType(0), _sft(0)
{
}


C_functor_definition::C_functor_definition(
   C_complex_functor_definition *c, SyntaxError * error)
   : C_production(error), _basis(_COMPLEX), _functor_category(0), 
     _declarator(0), _constructor_ptl(0), _complex_functor_def(c), 
     _c_script_def(0), _functorType(0), _sft(0)
{
}


C_functor_definition::C_functor_definition(
   C_connection_script_definition *c, SyntaxError * error)
   : C_production(error), _basis(_SCRIPT), _functor_category(0), 
     _declarator(0), _constructor_ptl(0), _complex_functor_def(0), 
     _c_script_def(c), _functorType(0), _sft(0)
{
}


C_functor_definition* C_functor_definition::duplicate() const
{
   return new C_functor_definition(*this);
}


void C_functor_definition::basicWork(LensContext *c)
{
   _declaratorName = _declarator->getName();
   _functorType = c->sim->getFunctorType(_declaratorName);
   if (_functorType==0) {
      std::string mes = "functor type \"" + _declarator->getName() + 
	 "\" does not exist";
      throwError(mes);
   }
   _categoryString = _functor_category->getCategory();
   std::string refCat(_functorType->getCategory());
   //   if (refCat!=_categoryString) {
   if (refCat.find(_categoryString)==std::string::npos) {
      std::string mes = "category mismatch, category of " + 
	 _declarator->getName() + " is " + refCat + ", not "  + 
	 _categoryString;
      throwError(mes);
   }

   // set up parameter type lists
   _constructor_list = &_empty;
   _function_list = &_empty;
   _return_list = &_empty;
}


void C_functor_definition::constrDefWork(LensContext *c)
{
   _declaratorName = _declarator->getName();
   _functorType = c->sim->getFunctorType(_declaratorName);
   if (_functorType==0) {
      std::string mes = "functor type " + _declarator->getName() + 
	 " does not exist";
      throwError(mes);
   }
   _categoryString = _functor_category->getCategory();
   std::string refCat(_functorType->getCategory());
   //if (refCat!=_categoryString) {
   if (refCat.find(_categoryString)==std::string::npos) {
      std::string mes = "category mismatch, category of " + 
	 _declarator->getName() + " is " + refCat + ", not "  + 
	 _categoryString;
      throwError(mes);
   }

   // set up parameter type lists
   _constructor_list = _constructor_ptl->getList();
   _function_list = &_empty;
   _return_list = &_empty;
}


void C_functor_definition::complexWork(LensContext *c)
{
   _declaratorName = _complex_functor_def->getName();
   _functorType = c->sim->getFunctorType(_declaratorName);
   if (_functorType==0) {
      std::string mes = "functor type " + _complex_functor_def->getName() + 
	 " does not exist";
      throwError(mes);
   }
   _categoryString = _complex_functor_def->getCategory();
   std::string refCat(_functorType->getCategory());
   //if (refCat!=_categoryString) {
   if (refCat.find(_categoryString)==std::string::npos) {
     std::string mes = "category mismatch, category of " + 
	 _complex_functor_def->getName() + " is " + refCat + ", not "  + 
	 _categoryString;
      throwError(mes);
   }

   // set up parameter type lists
   _function_list = _complex_functor_def->getFunctionParameters();
   _constructor_list =_complex_functor_def->getConstructorParameters();
   _return_list = _complex_functor_def->getReturnParameters();
}


void C_functor_definition::scriptWork(LensContext *c)
{
   _declaratorName = _c_script_def->getScriptName();
   _sft = new ScriptFunctorType(_c_script_def,_declaratorName);
   _functorType = _sft;
   _categoryString = _sft->getCategory();

   // set up parameter type lists
   _constructor_list = &_empty;
   _function_list = _c_script_def->getFunctionParameters();
   _return_list = &_empty;
}


C_functor_definition::~C_functor_definition()
{
   delete _functor_category;
   delete _constructor_ptl;
   delete _declarator;
   delete _complex_functor_def;
   delete _c_script_def;
   delete _sft;
}

void C_functor_definition::checkChildren() 
{
   if (_functor_category) {
      _functor_category->checkChildren();
      if (_functor_category->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_constructor_ptl) {
      _constructor_ptl->checkChildren();
      if (_constructor_ptl->isError()) {
         setError();
      }
   }
   if (_complex_functor_def) {
      _complex_functor_def->checkChildren();
      if (_complex_functor_def->isError()) {
         setError();
      }
   }
   if (_c_script_def) {
      _c_script_def->checkChildren();
      if (_c_script_def->isError()) {
         setError();
      }
   }
} 

void C_functor_definition::recursivePrint() 
{
   if (_functor_category) {
      _functor_category->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_constructor_ptl) {
      _constructor_ptl->recursivePrint();
   }
   if (_complex_functor_def) {
      _complex_functor_def->recursivePrint();
   }
   if (_c_script_def) {
      _c_script_def->recursivePrint();
   }
   printErrorMessage();
} 
