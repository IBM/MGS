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

#ifndef Connection_H
#define Connection_H
#include "Mdl.h"

#include "MemberContainer.h"
#include "InterfaceToMember.h"
#include "PSetToMember.h"
#include "Class.h"
#include <memory>
#include <string>
#include <set>

class Predicate;
class ConnectionCCBase;

class Connection {

   public:
      enum ComponentType {_EDGE, _NODE, _CONSTANT, _VARIABLE};
      enum DirectionType {_PRE, _POST};
      Connection(DirectionType directionType = _PRE,
		 ComponentType componentType = _EDGE, 		 
		 bool graph = false);
      virtual void duplicate(std::auto_ptr<Connection>& rv) const = 0;
      virtual ~Connection();
      bool getGraph() const;
      void setGraph(bool graph);

      std::string getString() const;

      void addInterfaceToMember(std::auto_ptr<InterfaceToMember>& im);

      const MemberContainer<InterfaceToMember>& getInterfaces() {
	 return _interfaces;
      }

      std::set<std::string> getInterfaceCasts(
	 const std::string& componentName) const;
      std::set<std::string> getInterfaceNames() const;
      virtual std::string getConnectionCode(
	 const std::string& name, 
	 const std::string& functionParameters) const = 0;

      void addInterfaceHeaders(Class& instance) const;

      ComponentType getComponentType() const {
	 return _componentType;
      };

      void setComponentType(ComponentType componentType) {
	 _componentType = componentType;
      }

      DirectionType getDirectionType() const {
	 return _directionType;
      };

      void setDirectionType(DirectionType directionType) {
	 _directionType = directionType;
      }
      
      PSetToMember& getPSetMappings() {
	 return _psetMappings;
      }

      void setPSetMappingsPSet(ConnectionCCBase* cc);

      std::string getStringForComponentType() const;
      std::string getStringForDirectionType() const;

      static std::string getStringForComponentType(ComponentType type);
      static std::string getParameterNameForComponentType(ComponentType type);
      static std::string getParametersForComponentType(ComponentType type);
      static std::string getStringForDirectionType(DirectionType type);
      static std::string getParametersForDirectionType(DirectionType type);

      void addMappingToInterface(
	 const std::string& interface, const std::string& interfaceMember,
	 const std::string& typeStr, std::auto_ptr<DataType>& dtToInsert);

   protected:
      MemberContainer<InterfaceToMember> _interfaces;    
      PSetToMember _psetMappings;
      
      virtual std::string getPredicateString() const {
	 return "";
      };

      virtual std::string getUserFunctionCallsString() const {
	 return "";
      };

      std::string getTypeString() const;

      std::string getCommonConnectionCode(const std::string& tab, 
					  const std::string& name) const;

      std::string getCommonConnectionCodeWithPredicate(const std::string& tab, 
					  const std::string& name) const;
      DirectionType _directionType;
      ComponentType _componentType;

   private:
      bool _graph;
};


#endif // Connection_H
