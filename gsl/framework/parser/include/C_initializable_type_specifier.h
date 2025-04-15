// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_initializable_type_specifier_H
#define C_initializable_type_specifier_H
#include "Copyright.h"

#include "C_type_specifier.h"
#include "C_production.h"

class C_parameter_type_pair;
class C_type_specifier;
class LensContext;
class SyntaxError;

class C_initializable_type_specifier : public C_production
{
   public:
      C_initializable_type_specifier(const C_initializable_type_specifier&);
      C_initializable_type_specifier(C_type_specifier::Type, SyntaxError *);
      C_initializable_type_specifier(C_type_specifier::Type, 
				     C_parameter_type_pair *, 
				     SyntaxError *);
      C_initializable_type_specifier(C_type_specifier::Type, 
				     C_type_specifier *, 
				     SyntaxError *);
      virtual ~C_initializable_type_specifier();
      virtual C_initializable_type_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_type_specifier::Type getType() const {
	 return _type;
      }
      C_parameter_type_pair* getParameterTypePair() const {
	 return _parameterTypePair;
      }

      C_type_specifier* getTypeSpecifier() const {
	 return _typeSpec;
      }

      C_type_specifier* getNextTypeSpecifier() {
	 return _typeSpec;
      }

   private:
      C_type_specifier::Type _type;
      C_parameter_type_pair* _parameterTypePair;
      C_type_specifier* _typeSpec;
};
#endif
