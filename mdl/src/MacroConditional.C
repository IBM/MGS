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
   _extraTest = "";
}

std::string MacroConditional::getBeginning() const
{
   std::string macroStart("");
   if (_name == "") {
   } else {
      if (_negate_condition)
	 macroStart += "#if ! defined(" + _name + ")\n";
      else
	 macroStart += "#if defined(" + _name + ")\n";
   }
   if (! _extraTest.empty())
   {
	 macroStart += "#if " + _extraTest + "\n";
   }
   return macroStart;
}

std::string MacroConditional::getEnding() const
{
   std::string macroEnd("");
   if (! _extraTest.empty())
   {
      macroEnd += "#endif\n";
   }
   if (_name == "") {
   } else {
      macroEnd += "#endif\n";
   }
   return macroEnd;
}

