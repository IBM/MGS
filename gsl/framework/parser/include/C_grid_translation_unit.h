// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_translation_unit_H
#define C_grid_translation_unit_H
#include "Copyright.h"

#include "C_production_grid.h"

class C_grid_translation_declaration_list;
class LensContext;
class Grid;
class SyntaxError;

class C_grid_translation_unit : public C_production_grid
{
   public:
      C_grid_translation_unit(const C_grid_translation_unit&);
      C_grid_translation_unit(C_grid_translation_declaration_list *, 
			      SyntaxError *);
      virtual ~C_grid_translation_unit();
      virtual C_grid_translation_unit* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      void setTdError(SyntaxError *tdError) { 
	 _tdError = tdError; 
      }
      void printTdError();

   private:
      C_grid_translation_declaration_list* _gl;
      SyntaxError* _tdError;
};
#endif
