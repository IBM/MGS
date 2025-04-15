// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_init_attr_type_node_H
#define C_init_attr_type_node_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_init_attr_type_node : public C_production
{
   public:
      enum Type {_IN, _OUT, _NODEINIT};
      C_init_attr_type_node(const C_init_attr_type_node&);
      C_init_attr_type_node(int t, SyntaxError *);
      virtual ~C_init_attr_type_node();
      virtual C_init_attr_type_node* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Type getType() {
	 return _type;
      }
      std::string getModelType();

   private:
      Type _type;
};
#endif
