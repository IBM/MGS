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

#ifndef C_name_H
#define C_name_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_name;
class LensContext;
class SyntaxError;

class C_name : public C_production
{
   public:
      C_name(const C_name&);
      C_name(std::string *, SyntaxError *);
      virtual ~C_name();
      virtual C_name* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getName() {
	 return *_name;
      }

   private:
      std::string *_name;
};
#endif
