// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_CONSTANT_H
#define C_ARGUMENT_CONSTANT_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_constant;
class C_argument;
class GslContext;
class DataItem;
class IntDataItem;
class FloatDataItem;

class C_argument_constant: public C_argument
{
   public:
      C_argument_constant(const C_argument_constant&);
      C_argument_constant(C_constant *, SyntaxError *);
      virtual ~C_argument_constant();
      virtual C_argument_constant* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_constant *getConstant() { 
	 return _constant;
      }
      DataItem *getArgumentDataItem() const;

   private:
      C_constant* _constant;
      IntDataItem* _int_dataitem;
      FloatDataItem* _float_dataitem;
};
#endif
