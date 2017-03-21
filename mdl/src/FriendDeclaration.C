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

#include "FriendDeclaration.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <sstream>

FriendDeclaration::FriendDeclaration(const std::string& name,
				     const std::string& conditional)
   : _name(name)
{
   if (conditional != "") {
      _macroConditional.setName(conditional);
   }
}

FriendDeclaration::~FriendDeclaration()
{
}

std::string FriendDeclaration::getCodeString() const
{
   std::ostringstream os;
   os << _macroConditional.getBeginning() 
      << TAB << "friend class " << _name << ";\n"
      << _macroConditional.getEnding();
   return os.str();
}

