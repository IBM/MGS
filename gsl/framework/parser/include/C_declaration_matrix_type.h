// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_matrix_type_H
#define C_declaration_matrix_type_H
#include "Copyright.h"

#include "C_declaration.h"
class C_matrix_type_specifier;
class C_matrix_init_declarator;
class GslContext;
class SyntaxError;

#include <memory>
#include <map>
#include <vector>

class C_declaration_matrix_type : public C_declaration
{
   public:
      C_declaration_matrix_type(const C_declaration_matrix_type&);
      C_declaration_matrix_type(C_matrix_type_specifier *, 
				C_matrix_init_declarator *, 
				SyntaxError *);
      virtual C_declaration_matrix_type* duplicate() const;
      virtual ~C_declaration_matrix_type();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_matrix_type_specifier* _matrixTypeSpecifier;
      C_matrix_init_declarator* _matrixInitDeclarator;

};
#endif
