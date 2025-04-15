// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_STRING_H
#define C_ARGUMENT_STRING_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_argument;
class LensContext;
class DataItem;
class CustomStringDataItem;
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
      CustomStringDataItem* _str_dataitem;
      std::string* _string;
};
#endif
