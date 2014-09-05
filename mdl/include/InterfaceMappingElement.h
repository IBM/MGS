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

#ifndef InterfaceMappingElement_H
#define InterfaceMappingElement_H
#include "Mdl.h"

#include "DataType.h"
#include <memory>
#include <string>

class InterfaceMappingElement {

   public:
      InterfaceMappingElement(const std::string& name, 
			      std::auto_ptr<DataType>& type, 
			      const std::string& typeString,
			      bool amp = false);
      
      InterfaceMappingElement(const InterfaceMappingElement& rv);
      InterfaceMappingElement& operator=(const InterfaceMappingElement& rv);
      const std::string& getName() const {
	 return _name;
      }

      const DataType* getDataType() const {
	 return _type;
      }

      bool getNeedsAmpersand() const {
	 return _needsAmpersand;
      }

      const std::string& getTypeString() const {
	 return _typeString;
      }

      std::string getServiceNameCode(const std::string& tab) const;

   protected:
      void destructOwnedHeap();
      void copyOwnedHeap(const InterfaceMappingElement& rv);
      
      std::string _name;
      DataType *_type;
      bool _needsAmpersand;
      std::string _typeString;
};


#endif // InterfaceMapping_H
