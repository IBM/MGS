// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_system_call_H
#define C_system_call_H
#include "Copyright.h"

#include <string>

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_system_call : public C_production
{
   public:
      C_system_call(std::string *, SyntaxError *);
      C_system_call(const C_system_call&);
      virtual ~C_system_call();
      virtual C_system_call* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::string* _command;
};
#endif
