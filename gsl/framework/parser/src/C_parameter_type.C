// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_parameter_type.h"
#include "C_type_specifier.h"
#include "C_matrix_type_specifier.h"
#include "C_functor_category.h"
#include "SyntaxError.h"
#include "C_production.h"

std::string const &C_parameter_type::getFunctorCategory()
{
   return _functor_category->getCategory();
}


void C_parameter_type::internalExecute(LensContext *c)
{
   if(_type_specifier) _type_specifier->execute(c);
   if(_matrix_type_spec) _matrix_type_spec->execute(c);
   if(_functor_category) _functor_category->execute(c);
}


C_parameter_type::C_parameter_type(const C_parameter_type& rv)
   : C_production(rv), _functor_category(0), _matrix_type_spec(0),
     _specified(rv._specified), _type(rv._type), _type_specifier(0), 
     _identifier(0)

{
   if ((rv._type == _TYPE_SPEC) && rv._type_specifier) {
      _type_specifier = rv._type_specifier->duplicate();
   }
   if ((rv._type == _MATRIX_TYPE_SPEC) && rv._matrix_type_spec) {
      _matrix_type_spec = rv._matrix_type_spec->duplicate();
   }
   if ((rv._type == _FUNCTOR_CAT) && rv._functor_category) {
      _functor_category = rv._functor_category->duplicate();
   }
   if (rv._identifier) {
      _identifier = new std::string(*(rv._identifier));
   }
}

C_parameter_type::C_parameter_type(SyntaxError * error)
   : C_production(error), _functor_category(0), _matrix_type_spec(0), 
     _specified(true), _type(_NULL), _type_specifier(0), _identifier(0)
{
   _identifier = new std::string("");
}


C_parameter_type::C_parameter_type(bool b, SyntaxError * error)
   : C_production(error), _functor_category(0), _matrix_type_spec(0), 
     _specified(b), _type(_UNSPECIFIED), _type_specifier(0), _identifier(0)
{
   _identifier = new std::string("");
}


C_parameter_type::C_parameter_type(C_type_specifier *t, std::string *id, 
				   SyntaxError * error)
   : C_production(error), _functor_category(0), _matrix_type_spec(0),
     _specified(true), _type(_TYPE_SPEC), _type_specifier(t), _identifier(id)
{
}


C_parameter_type::C_parameter_type(C_matrix_type_specifier *m, std::string *id,
				   SyntaxError * error)
   : C_production(error), _functor_category(0), _matrix_type_spec(m), 
     _specified(true), _type(_MATRIX_TYPE_SPEC), _type_specifier(0), 
     _identifier(id)
{
}


C_parameter_type::C_parameter_type(C_functor_category *f, std::string *id, 
				   SyntaxError * error)
   : C_production(error), _functor_category(f), _matrix_type_spec(0), 
     _specified(true), _type(_FUNCTOR_CAT), _type_specifier(0), _identifier(id)
{
}


C_parameter_type* C_parameter_type::duplicate() const
{
   return new C_parameter_type(*this);
}


C_parameter_type::~C_parameter_type()
{
   delete _type_specifier;
   delete _matrix_type_spec;
   delete _functor_category;
   delete _identifier;
}

void C_parameter_type::checkChildren() 
{
   if (_functor_category) {
      _functor_category->checkChildren();
      if (_functor_category->isError()) {
         setError();
      }
   }
   if (_matrix_type_spec) {
      _matrix_type_spec->checkChildren();
      if (_matrix_type_spec->isError()) {
         setError();
      }
   }
   if (_type_specifier) {
      _type_specifier->checkChildren();
      if (_type_specifier->isError()) {
         setError();
      }
   }
} 

void C_parameter_type::recursivePrint() 
{
   if (_functor_category) {
      _functor_category->recursivePrint();
   }
   if (_matrix_type_spec) {
      _matrix_type_spec->recursivePrint();
   }
   if (_type_specifier) {
      _type_specifier->recursivePrint();
   }
   printErrorMessage();
} 
