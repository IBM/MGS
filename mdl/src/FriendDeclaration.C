// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

