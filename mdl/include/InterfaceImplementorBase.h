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

#ifndef InterfaceImplementorBase_H
#define InterfaceImplementorBase_H
#include "Mdl.h"

#include "Generatable.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "MemberToInterface.h"
#include "Constants.h"
#include "Connection.h"
#include "Interface.h"
#include "Phase.h"

#include <memory>
#include <string>
#include <set>

class StructType;

class InterfaceImplementorBase : public Generatable {

   public:
      InterfaceImplementorBase(const std::string& fileName);
      InterfaceImplementorBase(const InterfaceImplementorBase& rv);
      InterfaceImplementorBase& operator=(const InterfaceImplementorBase& rv);
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const =0;
      virtual void generate() const;
      virtual std::string generateExtra() const;
      virtual std::string getType() const =0;
      virtual void setInstancePhases(std::auto_ptr<std::vector<Phase*> >& phases);
      virtual ~InterfaceImplementorBase();
      void releaseOutAttrPSet(std::auto_ptr<StructType>& oap);
      void setOutAttrPSet(std::auto_ptr<StructType>& oap);
      void checkAllMemberToInterfaces();

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name) {
	 _name = name;
      }

      StructType* getOutAttrPSet() {
	 return _outAttrPSet;
      }

      void addDataTypeToInstances(std::auto_ptr<DataType>& dataType) {
	 checkInstanceVariableNameSpace(dataType->getName());
	 _instances.addMember(dataType->getName(), dataType);
      }

      const MemberContainer<DataType>& getInstances() const {
	 return _instances;
      }

      void addMemberToInterfaceMapping(std::auto_ptr<MemberToInterface>& mti) {
	 _interfaces.addMemberToFront(mti->getInterface()->getName(), mti);
      }

      void addMappingToInterface(
	 const std::string& interfaceName, const std::string& interfaceMemberName, 
	 std::auto_ptr<DataType>& dtToInsert, bool ampersand);

      const MemberContainer<MemberToInterface>& getInterfaces() const {
	 return _interfaces;
      }

      void addDataTypeToOptionalServices(std::auto_ptr<DataType>& dataType) {
	 checkInstanceVariableNameSpace(dataType->getName());
	 _optionalInstanceServices.addMember(dataType->getName(), dataType);
      }

      const MemberContainer<DataType>& getOptionalInstanceServices() const {
	 return _optionalInstanceServices;
      }

      std::string getCommonPSetName(const std::string& type) const {
	 return PREFIX + getName() + type + "PSet";
      }
      std::string getInAttrPSetName() const {
	 return getCommonPSetName("InAttr");
      }
      std::string getOutAttrPSetName() const {
	 return getCommonPSetName("OutAttr");
      }
      std::string getPublisherName() const {
	 return PREFIX + getName() + "Publisher";
      }
      std::string getInstanceBaseName() const {
	 return PREFIX + getName();
      }

      std::string getRelationalDataUnitName() const {
	 return getType() + "RelationalDataUnit";
      }

      std::string getInstanceProxyName() const {
	 return PREFIX + getName() + "Proxy";
      }

      std::string getInstanceProxyDemarshallerName() const {
	 return PREFIX + getName() + "ProxyDemarshaller";
      }

      bool isMemberToInterface(const DataType& member) const;

      void setInterfaceImplementors();

   protected:
      virtual std::string getModuleName() const;

      void createPSetClass(std::auto_ptr<Class>& instance
			   , const MemberContainer<DataType>& members
			   , const std::string& name = "") const; 
      void generateOutAttrPSet();
      void generatePublisher();
      std::unique_ptr<Class> generateInstanceBase(); 
      std::unique_ptr<Class> generateInstanceBase(Class::PrimeType type); //to support GPU, the CG_LifeNode class's data is now part of CG_LifeNodeCompCategory, so we expand the API to return the Class object representing CG_LifeNodeCompCategory
      //std::unique_ptr<Class> generateInstanceBase(const Class::PrimeType& type = Class::PrimeType::UN_SET); //to support GPU, the CG_LifeNode class's data is now part of CG_LifeNodeCompCategory, so we expand the API to return the Class object representing CG_LifeNodeCompCategory
      void generateInstanceProxy();
      
      void addInstanceServiceHeaders(std::auto_ptr<Class>& instance) const;
      void addOptionalInstanceServiceHeaders(
	 std::auto_ptr<Class>& instance) const;
      virtual void addExtraServiceHeaders(
	 std::auto_ptr<Class>& instance) const;
      virtual void addExtraOptionalServiceHeaders(
	 std::auto_ptr<Class>& instance) const;
      std::string getInstanceServices(const std::string& tab) const;
      std::string getOptionalInstanceServices(const std::string& tab) const;
      virtual std::string getExtraServices(const std::string& tab) const;
      virtual std::string getExtraOptionalServices(
	 const std::string& tab) const;

      std::string getInstanceServiceNames(const std::string& tab, 
	    MachineType mach_type=MachineType::CPU) const;
      std::string getOptionalInstanceServiceNames(
	 const std::string& tab) const;
      virtual std::string getExtraServiceNames(const std::string& tab) const;
      virtual std::string getExtraOptionalServiceNames(
	 const std::string& tab) const;

      std::string getInstanceServiceDescriptions(
	 const std::string& tab,
	 MachineType mach_type=MachineType::CPU
	 ) const;
      std::string getOptionalInstanceServiceDescriptions(
	 const std::string& tab) const;
      virtual std::string getExtraServiceDescriptions(
	 const std::string& tab) const;
      virtual std::string getExtraOptionalServiceDescriptions(
	 const std::string& tab) const;

      std::string getInstanceServiceDescriptors(
	 const std::string& tab) const;
      std::string getOptionalInstanceServiceDescriptors(
	 const std::string& tab) const;
      virtual std::string getExtraServiceDescriptors(
	 const std::string& tab) const;
      virtual std::string getExtraOptionalServiceDescriptors(
	 const std::string& tab) const;

      std::string createServices(const MemberContainer<DataType>& members, 
				 const std::string& tab) const;

      std::string createOptionalServices(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab) const;

      std::string createServiceNames(const MemberContainer<DataType>& members, 
				     const std::string& tab,
				     MachineType mach_type=MachineType::CPU
				     ) const;

      std::string createOptionalServiceNames(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab) const;

      std::string createServiceDescriptors(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab) const;

      std::string createOptionalServiceDescriptors(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab) const;

      std::string createServiceDescriptions(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab,
	 MachineType mach_type=MachineType::CPU
	 ) const;

      std::string createOptionalServiceDescriptions(
	 const MemberContainer<DataType>& members, 
	 const std::string& tab) const;

      void setupInstanceInterfaces(std::auto_ptr<Class>& instance);

      void setupProxyInterfaces(std::auto_ptr<Class>& instance);

      // Will be implemented in derived classes.
      virtual void setupExtraInterfaces(std::auto_ptr<Class>& instance) {
	 return;
      }

      virtual std::string getAddPostEdgeFunctionBody() const;
      virtual std::string getAddPostNodeFunctionBody() const;
      virtual std::string getAddPostVariableFunctionBody() const;

      // Will be implemented in derived classes.
      virtual void addExtraInstanceBaseMethods(Class& instance) const {
	 return;
      }

      // Will be implemented in derived classes.
      virtual void addExtraInstanceMethods(Class& instance) const {
	 return;
      }

      // Will be implemented in derived classes.
      virtual void addExtraInstanceProxyMethods(Class& instance) const {
	 return;
      }

      std::string getAddConnectionFunctionBody(
	 Connection::ComponentType componentType, 
	 Connection::DirectionType directionType) const;

      // Will be implemented in derived classes.
      virtual std::string getAddConnectionFunctionBodyExtra(
	 Connection::ComponentType componentType, 
	 Connection::DirectionType directionType,
	 const std::string& componentName, const std::string& psetType, 
	 const std::string& psetName) const {
	 return "";
      }

      void addGetPublisherMethod(Class& instance) const;

      virtual void checkInstanceVariableNameSpace(const std::string& name) const;
      
      virtual unsigned getExtraNumberOfServices() const {
	 return 0;
      }

      void addDistributionCodeToIB(Class& instance);      

      const std::vector<DataType*>& getInterfaceImplementors() const {
	 return _interfaceImplementors;
      }

      virtual std::string getCompCategoryBaseName() const {
	 //NOTE: must be re-implemented by the derived class which is a CompCategoryBase
	 return "";
      };

      std::vector<Phase*>* _instancePhases;

   private:
      std::string getServiceNameCode() const;

      void copyOwnedHeap(const InterfaceImplementorBase& rv);
      void destructOwnedHeap();
      MemberContainer<DataType> _instances;
      MemberContainer<MemberToInterface> _interfaces;
      std::vector<DataType*> _interfaceImplementors;
      std::string _name;
      StructType* _outAttrPSet;
      MemberContainer<DataType> _optionalInstanceServices;
};


#endif // InterfaceImplementorBase_H
