// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_NODESET_H
#define C_ARGUMENT_NODESET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_nodeset;
class NodeSet;
class DataItem;
class NodeSetDataItem;
class C_argument;
class GslContext;
class C_gridset;
class SyntaxError;

class C_argument_nodeset: public C_argument
{
   public:
      C_argument_nodeset(const C_argument_nodeset&);
      C_argument_nodeset(C_nodeset *, SyntaxError *);
      C_argument_nodeset(C_gridset *, SyntaxError *);
      virtual ~C_argument_nodeset();
      virtual C_argument_nodeset* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_nodeset *getNodeset(){
	 return _nodeset;
      }
      DataItem* getArgumentDataItem() const;

   private:
      C_nodeset* _nodeset;
      C_gridset* _gridset;
      NodeSetDataItem* _nodeset_DI;
};
#endif
