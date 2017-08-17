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
