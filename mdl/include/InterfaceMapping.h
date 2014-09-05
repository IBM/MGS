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

#ifndef InterfaceMapping_H
#define InterfaceMapping_H
#include "Mdl.h"

#include <memory>
#include <vector>
#include "DataType.h"
#include "InterfaceMappingElement.h"

class Interface;

class InterfaceMapping {

   public:
      typedef std::vector<InterfaceMappingElement>::const_iterator 
      const_iterator;
      typedef std::vector<InterfaceMappingElement>::iterator iterator;

      InterfaceMapping(Interface* interface = 0);
      virtual void duplicate(std::auto_ptr<InterfaceMapping>& rv) const = 0;
      virtual ~InterfaceMapping();
      void setInterface(Interface* interface) {
	 _interface = interface;
      }
      const Interface* getInterface() const {
	 return _interface;
      }
      void addMapping(const std::string& name, std::auto_ptr<DataType>& data, 
		      bool amp = false);

      std::vector<InterfaceMappingElement>& getMappings() {
	 return _mappings;
      }

      inline const_iterator begin() const {
	 return _mappings.begin();
      }
      inline const_iterator end() const {
	 return _mappings.end();
      }
      inline iterator begin() {
	 return _mappings.begin();
      }
      inline iterator end() {
	 return _mappings.end();
      }
      iterator find(const std::string& token);

   protected:
      virtual void checkAndExtraWork(const std::string& name,
	 DataType* member, const DataType* interface, bool amp) = 0;

      bool existsInMappings(const std::string& token) const;

      std::string commonGenerateString(const std::string& interfaceName, 
				       const std::string& direction,
				       const std::string& tab) const;

      Interface* _interface;
      std::vector<InterfaceMappingElement> _mappings;
};


#endif // InterfaceMapping_H
