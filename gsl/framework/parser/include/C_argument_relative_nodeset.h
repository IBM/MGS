// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_RELATIVE_NODESET_H
#define C_ARGUMENT_RELATIVE_NODESET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_relative_nodeset;
class C_argument;
class LensContext;
class DataItem;
class RelativeNodeSetDataItem;
class SyntaxError;

class C_argument_relative_nodeset: public C_argument
{
   public:
      C_argument_relative_nodeset(const C_argument_relative_nodeset&);
      C_argument_relative_nodeset(C_relative_nodeset *, SyntaxError *);
      virtual ~C_argument_relative_nodeset();
      virtual C_argument_relative_nodeset* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_relative_nodeset *getRelative_nodeset() {
	 return _rel_nodeset;
      }
      DataItem *getArgumentDataItem() const;

   private:
      C_relative_nodeset* _rel_nodeset;
      RelativeNodeSetDataItem* _rel_nodeset_DI;
};
#endif
