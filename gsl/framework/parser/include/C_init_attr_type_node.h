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
