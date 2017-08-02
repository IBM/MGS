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

#ifndef C_layer_name_H
#define C_layer_name_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_layer_name : public C_production
{
   public:
      C_layer_name(const C_layer_name&);
      C_layer_name(std::string *, SyntaxError *);
      virtual ~C_layer_name();
      virtual C_layer_name* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string& getName() {
	 return *_name;
      }

   private:
      std::string* _name;

};
#endif
