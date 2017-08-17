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

#ifndef C_production_H
#define C_production_H
#include "Copyright.h"

#include <string>

class LensContext;
class SyntaxError;

class C_production
{
   public:
      C_production(SyntaxError* error);
      C_production(const C_production&);
      virtual ~C_production ();
      virtual C_production* duplicate() const = 0;
      bool isError();
      void setError();
      void printErrorMessage();
      virtual void throwError(const std::string&);
      virtual void execute(LensContext *);
      virtual void checkChildren() {};
      virtual void recursivePrint() {};
   protected:
      virtual void internalExecute(LensContext *) = 0;
      SyntaxError *_error;
};
#endif
