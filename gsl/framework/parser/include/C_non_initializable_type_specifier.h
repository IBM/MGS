// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_non_initializable_type_specifier_H
#define C_non_initializable_type_specifier_H
#include "Copyright.h"

#include "C_type_specifier.h"
#include "C_production.h"

class GslContext;
class SyntaxError;

class C_non_initializable_type_specifier : public C_production
{
   public:
      C_non_initializable_type_specifier(
	 const C_non_initializable_type_specifier&);
      C_non_initializable_type_specifier(
	 C_type_specifier::Type, SyntaxError *);
      virtual ~C_non_initializable_type_specifier ();
      virtual C_non_initializable_type_specifier* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_type_specifier::Type getType() const { 
	 return _type;
      }
      C_type_specifier* getNextTypeSpecifier() {
	 return _nextTypeSpec;
      }

   private:
      C_type_specifier::Type _type;
      C_type_specifier* _nextTypeSpec;
};
#endif
