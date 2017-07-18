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

#ifndef C_ARGUMENT_GRID_GRANULE_H
#define C_ARGUMENT_GRID_GRANULE_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_grid_granule;
class DataItem;
class GranuleMapperDataItem;
class C_argument;
class LensContext;
class SyntaxError;

class C_argument_grid_granule: public C_argument
{
   public:
      C_argument_grid_granule(const C_argument_grid_granule&);
      C_argument_grid_granule(C_grid_granule *, SyntaxError *);
      virtual ~C_argument_grid_granule();
      virtual C_argument_grid_granule* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_grid_granule *getGrid_Granule() { 
	 return _granuleMapper;
      }
      DataItem* getArgumentDataItem() const;

   private:
      C_grid_granule* _granuleMapper;
      GranuleMapperDataItem* _granuleMapperDI;
};
#endif
