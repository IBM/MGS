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

#ifndef C_ARGUMENT_CONSTANT_H
#define C_ARGUMENT_CONSTANT_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_constant;
class C_argument;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
