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

#ifndef C_declaration_matrix_type_H
#define C_declaration_matrix_type_H
#include "Copyright.h"

#include "C_declaration.h"
class C_matrix_type_specifier;
class C_matrix_init_declarator;
class LensContext;
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
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_matrix_type_specifier* _matrixTypeSpecifier;
      C_matrix_init_declarator* _matrixInitDeclarator;

};
#endif
