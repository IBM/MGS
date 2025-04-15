// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_translation_declaration_H
#define C_grid_translation_declaration_H
#include "Copyright.h"

#include "C_production_grid.h"

class C_declaration;
class C_grid_function_specifier;
class LensContext;
class Grid;
class SyntaxError;

class C_grid_translation_declaration : public C_production_grid
{
   public:
      C_grid_translation_declaration(const C_grid_translation_declaration&);
      C_grid_translation_declaration(C_declaration *, SyntaxError *);
      C_grid_translation_declaration(C_grid_function_specifier *, 
				     SyntaxError *);
      virtual ~C_grid_translation_declaration();
      virtual C_grid_translation_declaration* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declaration* _declaration;
      C_grid_function_specifier* _gridFuncSpec;

};
#endif
