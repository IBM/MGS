// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TypeDefinition.h"
#include "Constants.h"
#include <sstream>

TypeDefinition::TypeDefinition(AccessType accessType)
   : _accessType(accessType), _definition("")
{
}
 
void TypeDefinition::printTypeDef(
   AccessType type, std::ostringstream& os) const
{
   if (type == _accessType) {
      os << _macroConditional.getBeginning() 
	 << TAB << TAB << "typedef " << _definition << ";\n" 
	 << _macroConditional.getEnding();
   } 
}
