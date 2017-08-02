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
