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

#ifndef C_definition_type_H
#define C_definition_type_H
#include "Copyright.h"

#include "C_definition.h"

class C_type_definition;
class LensContext;
class SyntaxError;

class C_definition_type : public C_definition
{
   public:
      C_definition_type(const C_definition_type&);
      C_definition_type(C_type_definition *, SyntaxError *);
      virtual C_definition_type* duplicate() const;
      virtual ~C_definition_type();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_type_definition* _typeDefinition;
};
#endif
