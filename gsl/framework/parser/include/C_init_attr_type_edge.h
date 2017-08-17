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

#ifndef C_init_attr_type_edge_H
#define C_init_attr_type_edge_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_init_attr_type_edge : public C_production
{
   public:
      C_init_attr_type_edge(const C_init_attr_type_edge&);
      C_init_attr_type_edge(SyntaxError *);
      virtual ~C_init_attr_type_edge();
      virtual C_init_attr_type_edge* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

};
#endif
