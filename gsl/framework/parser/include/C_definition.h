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

#ifndef C_DEFINITION_H
#define C_DEFINITION_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_definition : public C_production
{
   public:
      C_definition(SyntaxError* error);
      C_definition(const C_definition&);
      virtual ~C_definition();
      virtual C_definition* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

};
#endif
