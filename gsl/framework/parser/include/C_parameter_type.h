// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_parameter_type_H
#define C_parameter_type_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_type_specifier;
class C_matrix_type_specifier;
class C_functor_category;
class GslContext;
class SyntaxError;

class C_parameter_type : public C_production
{
   public:
      enum Type
      {
         _TYPE_SPEC,
         _MATRIX_TYPE_SPEC,
         _FUNCTOR_CAT,
         _NULL,
         _UNSPECIFIED
      };
      C_parameter_type(C_parameter_type const&);
      C_parameter_type(SyntaxError *);
      C_parameter_type(bool, SyntaxError *);
      C_parameter_type(C_type_specifier *, std::string *id, SyntaxError *);
      C_parameter_type(C_matrix_type_specifier *, std::string *id, 
		       SyntaxError *);
      C_parameter_type(C_functor_category *, std::string *id, SyntaxError *);
      virtual ~C_parameter_type ();
      virtual C_parameter_type* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      bool isSpecified() const {
	 return _specified;
      }
      Type getType() const { 
	 return _type;
      }
      const std::string& getIdentifier() const { 
	 return *_identifier;
      }
      C_type_specifier* getTypeSpecifier() {
	 return _type_specifier;
      }
      C_matrix_type_specifier* getMatrixTypeSpecifier() { 
	 return _matrix_type_spec;
      }
      const std::string& getFunctorCategory();

   private:
      C_functor_category* _functor_category;
      C_matrix_type_specifier* _matrix_type_spec;
      bool _specified;
      Type _type;
      C_type_specifier* _type_specifier;
      std::string* _identifier;
};
#endif
