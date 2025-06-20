// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_DECLARATOR_H
#define C_ARGUMENT_DECLARATOR_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_declarator;
class C_argument;
class GslContext;
class DataItem;
class SyntaxError;

class C_argument_declarator: public C_argument
{
   public:
      C_argument_declarator(const C_argument_declarator&);
      C_argument_declarator(C_declarator *, SyntaxError *);
      virtual ~C_argument_declarator();
      virtual C_argument_declarator* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_declarator *getDeclarator() { 
	 return _declarator;
      }
      DataItem *getArgumentDataItem() const {
	 return _dataitem;
      }

   private:
      DataItem* _dataitem;
      C_declarator* _declarator;
};
#endif
