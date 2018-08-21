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

#ifndef C_value_H
#define C_value_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_value : public C_production
{
   public:
      C_value(const C_value& rv);
      C_value(std::string *, SyntaxError *);
      virtual ~C_value();
      virtual C_value* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getValue() {
	 return *_value;
      }

   private:
      std::string* _value;
};
#endif
