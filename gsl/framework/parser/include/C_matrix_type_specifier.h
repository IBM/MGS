// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_matrix_type_specifier_H
#define C_matrix_type_specifier_H
#include "Copyright.h"

#include "C_production.h"

class C_type_specifier;
class GslContext;
class SyntaxError;

class C_matrix_type_specifier: public C_production
{
   public:
      C_matrix_type_specifier(const C_matrix_type_specifier&);
      C_matrix_type_specifier(C_type_specifier *, SyntaxError *);
      virtual ~C_matrix_type_specifier();
      virtual C_matrix_type_specifier* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_type_specifier* getTypeSpecifier() const;

   private:
      C_type_specifier* _typeSpecifier;
};
#endif
