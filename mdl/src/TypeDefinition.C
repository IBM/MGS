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

#include "TypeDefinition.h"
#include "Constants.h"
#include <sstream>

TypeDefinition::TypeDefinition(int accessType)
   : _accessType(accessType), _definition("")
{
}
 
void TypeDefinition::printTypeDef(
   int type, std::ostringstream& os) const
{
   if (type == _accessType) {
      os << _macroConditional.getBeginning() 
	 << TAB << TAB << "typedef " << _definition << ";\n" 
	 << _macroConditional.getEnding();
   } 
}
