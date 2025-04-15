// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_VOID_H
#define C_ARGUMENT_VOID_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_argument;
class LensContext;
class DataItem;
class SyntaxError;

class C_argument_void: public C_argument
{
   public:
      C_argument_void(const C_argument_void&);
      C_argument_void(SyntaxError *);
      virtual ~C_argument_void ();
      virtual C_argument_void* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      DataItem *getArgumentDataItem() const;

   private:
};
#endif
