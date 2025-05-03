// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_GRIDSET_H
#define C_ARGUMENT_GRIDSET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_gridset;
class C_argument;
class GslContext;
class DataItem;
class GridSetDataItem;
class SyntaxError;

class C_argument_gridset: public C_argument
{
   public:
      C_argument_gridset(const C_argument_gridset&);
      C_argument_gridset(C_gridset *, SyntaxError *);
      virtual ~C_argument_gridset ();
      virtual C_argument_gridset* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_gridset* getGridset() { 
	 return _gridset; 
      }
      DataItem *getArgumentDataItem() const;

   private:
      C_gridset* _gridset;
      GridSetDataItem* _dataitem;
};
#endif
