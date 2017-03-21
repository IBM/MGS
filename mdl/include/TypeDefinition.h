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
      TypeDefinition(int accessType = AccessType::PUBLIC);
      ~TypeDefinition() {};

      int getAccessType() const {
	 return _accessType;
      }

      void setAccessType(int acc) {
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

      void printTypeDef(int type, std::ostringstream& os) const;

   private:
      // Shows if the attribute is public, protected, or private
      int _accessType;

      // The definition defined by the user
      std::string _definition;

      MacroConditional _macroConditional;
};

#endif
