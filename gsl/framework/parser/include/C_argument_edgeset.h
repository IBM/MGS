// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
class GslContext;
class SyntaxError;

class C_argument_edgeset: public C_argument
{
   public:
      C_argument_edgeset(const C_argument_edgeset&);
      C_argument_edgeset(C_edgeset *, SyntaxError *);
      virtual ~C_argument_edgeset();
      virtual C_argument_edgeset* duplicate() const;
      virtual void internalExecute(GslContext *);
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
