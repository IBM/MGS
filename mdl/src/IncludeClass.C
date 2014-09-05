// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
