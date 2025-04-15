#include <memory>
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "InterfaceMappingElement.h"
#include "Constants.h"
#include <string>
#include <sstream>

InterfaceMappingElement::InterfaceMappingElement(
   const std::string& name, std::unique_ptr<DataType>&& type, 
   const std::string& typeString, bool amp)
   : _name(name), _type(0), _needsAmpersand(amp), _typeString(typeString)
{
   _type = type.release();
}

InterfaceMappingElement::InterfaceMappingElement(
   const InterfaceMappingElement& rv)
   : _name(rv._name), _needsAmpersand(rv._needsAmpersand), 
     _typeString(rv._typeString)
{
   copyOwnedHeap(rv);
}

InterfaceMappingElement& InterfaceMappingElement::operator=(
   const InterfaceMappingElement& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _name = rv._name;
      _needsAmpersand = rv._needsAmpersand;
      _typeString = rv._typeString;
   }
   return *this;
}

void InterfaceMappingElement::destructOwnedHeap()
{
   delete _type;
}

void InterfaceMappingElement::copyOwnedHeap(const InterfaceMappingElement& rv)
{
   if (rv._type) {
      std::unique_ptr<DataType> dup;
      rv._type->duplicate(std::move(dup));
      _type = dup.release();
   } else {
      _type = 0;
   }
}

std::string InterfaceMappingElement::getServiceNameCode(
   const std::string& tab) const
{
   std::ostringstream os;
   
   os << tab << "if (" << SUBINTERFACENAME << " == \"" << _name << "\") {\n"
      << tab << TAB << "return \"" << _type->getName() << "\";\n"
      << tab << "}\n";
   return os.str();
}
