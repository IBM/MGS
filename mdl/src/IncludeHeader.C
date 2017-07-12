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

#include "IncludeHeader.h"
#include "Constants.h"
#include <sstream>

IncludeHeader::IncludeHeader(const std::string& name, 
			     const std::string& conditional)
   : _name(name)
{
   if (conditional != "") {
      _macroConditional.setName(conditional);
   }
}
 
std::string IncludeHeader::getHeaderCode() const
{
   if (_name == "") {
      return "";
   }
   std::ostringstream os;
   os << _macroConditional.getBeginning() 
      << "#include " << _name << "\n" 
      << _macroConditional.getEnding();
   return os.str();
}
