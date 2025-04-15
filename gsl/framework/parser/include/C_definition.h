// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_DEFINITION_H
#define C_DEFINITION_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_definition : public C_production
{
   public:
      C_definition(SyntaxError* error);
      C_definition(const C_definition&);
      virtual ~C_definition();
      virtual C_definition* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

};
#endif
