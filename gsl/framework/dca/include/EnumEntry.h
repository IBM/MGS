// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ENUMENTRY_H
#define ENUMENTRY_H
#include "Copyright.h"

#include <string>


class EnumEntry
{

   public:
      EnumEntry(std::string value, std::string description);
      EnumEntry(EnumEntry*);
      std::string getValue();
      std::string getDescription();
      ~EnumEntry();

   private:
      std::string _value;
      std::string _description;
};
#endif
