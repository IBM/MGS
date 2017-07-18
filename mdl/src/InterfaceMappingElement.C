// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "InterfaceMappingElement.h"
#include "Constants.h"
#include <string>
#include <sstream>

InterfaceMappingElement::InterfaceMappingElement(
   const std::string& name, std::auto_ptr<DataType>& type, 
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
      std::auto_ptr<DataType> dup;
      rv._type->duplicate(dup);
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
