// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "IncludeClass.h"
#include "Constants.h"
#include <sstream>

IncludeClass::IncludeClass(const std::string& name, 
			   const std::string& conditional)
   : _name(name)
{
   if (conditional != "") {
      _macroConditional.setName(conditional);
   }
}
 
std::string IncludeClass::getClassCode() const
{
   if (_name == "") {
      return "";
   }
   std::ostringstream os;
   os << _macroConditional.getBeginning() 
      << "class " << _name << ";\n" 
      << _macroConditional.getEnding();
   return os.str();
}

std::string IncludeClass::getHeaderCode() const
{
   if (_name == "") {
      return "";
   }
   std::ostringstream os;
   os << _macroConditional.getBeginning() 
      << "#include \"" << _name << ".h\"\n"
      << _macroConditional.getEnding();
   return os.str();
}
