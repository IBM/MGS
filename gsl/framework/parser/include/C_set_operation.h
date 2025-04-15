// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_set_operation_H
#define C_set_operation_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_set_operation : public C_production
{
   public:
      enum Type {_SET, _COPY};

      C_set_operation(const C_set_operation&);
      C_set_operation(std::string, SyntaxError *);
      virtual ~C_set_operation();
      virtual C_set_operation* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      Type _type;
};
#endif
