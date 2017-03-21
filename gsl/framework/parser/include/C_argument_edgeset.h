// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_ARGUMENT_EDGESET_H
#define C_ARGUMENT_EDGESET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_edgeset;
class EdgeSet;
class DataItem;
class EdgeSetDataItem;
class C_argument;
class LensContext;
class SyntaxError;

class C_argument_edgeset: public C_argument
{
   public:
      C_argument_edgeset(const C_argument_edgeset&);
      C_argument_edgeset(C_edgeset *, SyntaxError *);
      virtual ~C_argument_edgeset();
      virtual C_argument_edgeset* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_edgeset *getEdgeSet() { 
	 return _edgeset;
      }
      DataItem* getArgumentDataItem() const;

   private:
      C_edgeset* _edgeset;
      EdgeSetDataItem* _edgesetDI;
};
#endif
