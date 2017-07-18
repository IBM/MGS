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

#ifndef C_ARGUMENT_STRING_H
#define C_ARGUMENT_STRING_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_argument;
class LensContext;
class DataItem;
class StringDataItem;
class SyntaxError;

class C_argument_string: public C_argument
{
   public:
      C_argument_string(const C_argument_string&);
      C_argument_string(std::string *, SyntaxError *);
      virtual ~C_argument_string();
      virtual C_argument_string* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      std::string& getString() {
	 return *_string;
      }
      DataItem* getArgumentDataItem() const;

   private:
      StringDataItem* _str_dataitem;
      std::string* _string;
};
#endif
