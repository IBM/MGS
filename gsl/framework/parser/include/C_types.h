// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_types_H
#define C_types_H
#include "Copyright.h"

#include "C_production.h"

class GslContext;
class SyntaxError;

class C_types : public C_production
{
   public:
	   //TUAN: these items are not used so far
	   //_NVPAIR, _MATRIX
	   //_FUNCTOR, _GRID, _COMPOSITE
      enum Type
      {
         _PSET,
         _REPNAME,
         _LIST,
         _MATRIX,
         _GRIDCOORD,
         _NVPAIR,
         _INT,
         _FLOAT,
         _STRING,
         _RELNODESET,
         _NODESET,
         _NODETYPE,
         _EDGETYPE,
         _FUNCTOR,
         _GRID,
         _COMPOSITE
      };
      C_types(SyntaxError *);
      C_types(const C_types& rv);
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_types* duplicate() const;
      virtual ~C_types();
      Type _type;
};
#endif
