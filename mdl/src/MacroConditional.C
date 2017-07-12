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

#include "MacroConditional.h"

#include <string>

MacroConditional::MacroConditional(const std::string& name)
   : _name(name)
{
}

std::string MacroConditional::getBeginning() const
{
   if (_name == "") {
      return "";
   } else {
      return "#ifdef " + _name + "\n";
   }
}

std::string MacroConditional::getEnding() const
{
   if (_name == "") {
      return "";
   } else {
      return "#endif\n";
   }
}

