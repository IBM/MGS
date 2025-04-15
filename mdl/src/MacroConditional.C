// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "MacroConditional.h"

#include <string>

MacroConditional::MacroConditional(const std::string& name)
#if ! defined(USE_COMPLEX_MACRO)
   : _name(name)
#endif
{
   _negate_condition = false;
   _extraTest = "";
#if defined(USE_COMPLEX_MACRO)
   _names.clear();
   _names.push_back(name);
#endif
}
#if defined(USE_COMPLEX_MACRO)
MacroConditional::MacroConditional(const std::vector<std::string>& names)
{
   _negate_condition = false;
   _extraTest = "";
   _names.clear();
   for (size_t ii = 0; ii < names.size(); ii++)
      _names.push_back(names[ii]);
}
#endif

#if defined(USE_COMPLEX_MACRO)
std::string MacroConditional::getBeginning() const
{
   std::string macroStart("");
   assert(_names.size() >= 1);
   if (_names[0] == "") {
   } else {
      if (_negate_condition)
      {
	 if (_names.size() == 1)
	    macroStart += "#if ! defined(" + _names[0] + ")\n";
	 else{
	    assert(0); //add implementation
	 }
      }
      else
      {
	 macroStart += "#if ";
	 for (size_t ii=0; ii < _names.size(); ii++)
	 {
	    if (ii > 0)
	       macroStart += " and ";
	    macroStart += " defined(" + _names[ii] + ")";
	 }
	 macroStart += "\n";
      }
   }
   if (! _extraTest.empty())
   {
	 macroStart += "#if " + _extraTest + "\n";
   }
   return macroStart;
}
#else
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
#endif

#if defined(USE_COMPLEX_MACRO)
std::string MacroConditional::getEnding() const
{ 
   std::string macroEnd("");
   if (! _extraTest.empty())
   {
      macroEnd += "#endif\n";
   }
   if (_names.size() == 1 and _names[0] == "") {
   } else {
      macroEnd += "#endif\n";
   }
   return macroEnd;
}
#else
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
#endif

