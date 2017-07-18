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

#ifndef C_set_operation_H
#define C_set_operation_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_set_operation : public C_production
{
   public:
      enum Type {_SET, _COPY};

      C_set_operation(const C_set_operation&);
      C_set_operation(std::string, SyntaxError *);
      virtual ~C_set_operation();
      virtual C_set_operation* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      Type _type;
};
#endif
