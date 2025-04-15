// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TypeDefinition_H
#define TypeDefinition_H
#include "Mdl.h"

#include <string>
#include <sstream>
#include "AccessType.h"
#include "MacroConditional.h"

// This class generates type definitions (code) to be used in classes.
class TypeDefinition
{
   public:
      TypeDefinition(AccessType accessType = AccessType::PUBLIC);
      ~TypeDefinition() {};

      AccessType getAccessType() const {
	 return _accessType;
      }

      void setAccessType(AccessType acc) {
	 _accessType = acc;
      }

      const std::string& getDefinition() const {
	 return _definition;
      }

      void setDefinition(const std::string& definition) {
	 _definition = definition;
      }

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      

      void printTypeDef(AccessType type, std::ostringstream& os) const;

   private:
      // Shows if the attribute is public, protected, or private
      AccessType _accessType;

      // The definition defined by the user
      std::string _definition;

      MacroConditional _macroConditional;
};

#endif
