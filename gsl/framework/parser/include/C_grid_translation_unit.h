// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
