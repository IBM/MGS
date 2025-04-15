// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_translation_declaration_list_H
#define C_grid_translation_declaration_list_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_grid_translation_declaration;
class LensContext;
class Grid;
class SyntaxError;

class C_grid_translation_declaration_list : public C_production_grid
{
   public:
      C_grid_translation_declaration_list(
	 const C_grid_translation_declaration_list&);
      C_grid_translation_declaration_list(
	 C_grid_translation_declaration *, SyntaxError *);
      C_grid_translation_declaration_list(
	 C_grid_translation_declaration_list *, 
	 C_grid_translation_declaration *, SyntaxError *);
      std::list<C_grid_translation_declaration*>* releaseList();
      virtual ~C_grid_translation_declaration_list();
      virtual C_grid_translation_declaration_list* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::list<C_grid_translation_declaration*>* _list;
};
#endif
