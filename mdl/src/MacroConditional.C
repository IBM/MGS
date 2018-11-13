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

#include "MacroConditional.h"

#include <string>

MacroConditional::MacroConditional(const std::string& name)
   : _name(name)
{
   _negate_condition = false;
}

std::string MacroConditional::getBeginning() const
{
   if (_name == "") {
      return "";
   } else {
      //return "#ifdef " + _name + "\n";
      if (_negate_condition)
	 return "#if ! defined( " + _name + ")\n";
      else
	 return "#if defined( " + _name + ")\n";
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

