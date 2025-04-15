// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_init_attr_type_edge_H
#define C_init_attr_type_edge_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_init_attr_type_edge : public C_production
{
   public:
      C_init_attr_type_edge(const C_init_attr_type_edge&);
      C_init_attr_type_edge(SyntaxError *);
      virtual ~C_init_attr_type_edge();
      virtual C_init_attr_type_edge* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

};
#endif
