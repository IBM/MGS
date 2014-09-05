// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_ARGUMENT_GRIDSET_H
#define C_ARGUMENT_GRIDSET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_gridset;
class C_argument;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
