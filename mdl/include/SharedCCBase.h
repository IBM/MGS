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

#ifndef SharedCCBase_H
#define SharedCCBase_H
#include "Mdl.h"

#include "ConnectionCCBase.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "Phase.h"
#include "TriggeredFunction.h"
#include "Class.h"
// #include "FinalPhase.h"
// #include "InitPhase.h"
// #include "LoadPhase.h"
// #include "RuntimePhase.h"
#include <memory>
#include <string>
#include <set>

class Generatable;

class SharedCCBase : public ConnectionCCBase {
   public:
      SharedCCBase(const std::string& fileName);
      SharedCCBase(const SharedCCBase& rv);
      SharedCCBase& operator=(const SharedCCBase& rv);
      virtual ~SharedCCBase();
      void addSharedPhase(std::unique_ptr<Phase>&& phase);
      void addSharedTriggeredFunction(
	 std::unique_ptr<TriggeredFunction>&& triggeredFunction);
      virtual std::string generateExtra() const;

      void addDataTypeToShareds(std::unique_ptr<DataType>&& dataType) {
	 checkInstanceVariableNameSpace(dataType->getName());
	 _shareds.addMemberToFront(dataType->getName(), std::move(dataType));
      }
      
      const MemberContainer<DataType>& getShareds() {
	 return _shareds;
      }

      void addDataTypeToOptioinalSharedServices(
	 std::unique_ptr<DataType>&& dataType) {
	 checkInstanceVariableNameSpace(dataType->getName());
	 _optionalSharedServices.addMemberToFront(dataType->getName(), std::move(dataType));
      }


      const MemberContainer<DataType>& getOptionalSharedServices() {
	 return _optionalSharedServices;
      }

   protected:
      // This method is used by generateFactory. It returns the name of the
      // class that will be loaded by the factory.
      virtual std::string getLoadedInstanceTypeName();

      // This method is used by generateFactory. It returns the arguments
      // that will be used  while instantiating the loaded class.
      virtual std::string getLoadedInstanceTypeArguments();

      void generateSharedMembers();
      void generateWorkUnitShared();

      std::string getSharedMembersName() const {
	 return PREFIX + getName() + "SharedMembers";
      }
      std::string getSharedMembersAttName() const {
	 return PREFIX + "sharedMembers";
      }
      std::string getWorkUnitSharedName() const {
	 return  getWorkUnitCommonName("Shared");
      }

      virtual void addExtraServiceHeaders(
	 std::unique_ptr<Class>&& instance) const;
      virtual void addExtraOptionalServiceHeaders(
	 std::unique_ptr<Class>&& instance) const;
      virtual std::string getExtraServices(const std::string& tab) const;
      virtual std::string getExtraOptionalServices(
	 const std::string& tab) const;

      virtual std::string getExtraServiceNames(const std::string& tab) const;
      virtual std::string getExtraOptionalServiceNames(
	 const std::string& tab) const;

      virtual std::string getExtraServiceDescriptions(
	 const std::string& tab) const;
      virtual std::string getExtraOptionalServiceDescriptions(
	 const std::string& tab) const;

      virtual std::string getExtraServiceDescriptors(
	 const std::string& tab) const;
      virtual std::string getExtraOptionalServiceDescriptors(
	 const std::string& tab) const;

      virtual std::string createGetWorkUnitsMethodBody(
	 const std::string& phaseName, const std::string& workUnits) const;

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
      virtual void addExtraInstanceMethods(Class& instance) const;
      virtual void addExtraInstanceProxyMethods(Class& instance) const;
      virtual void addExtraCompCategoryBaseMethods(Class& instance) const;
      virtual void addExtraCompCategoryMethods(Class& instance) const;
      virtual std::string getCompCategoryBaseConstructorBody() const;

      virtual void checkInstanceVariableNameSpace(const std::string& name) const;

      virtual unsigned getExtraNumberOfServices() const {
	 return _shareds.size() + _optionalSharedServices.size();
      }
      virtual void addSharedDataToKernelArgs(Class* c){
	 if (_shareds.size() > 0) {
	    MemberContainer<DataType>::const_iterator it, end = _shareds.end();
	    for (it = _shareds.begin(); it != end; ++it) {
	       //std::string name="getSharedMembers()." + it->first;
	       //c->addKernelArgs(name, it->first, it->second->getDescriptor());
	       c->addKernelArgs(it->second, true);
	    }
	 }
      }

   private:
      void copyOwnedHeap(const SharedCCBase& rv);
      void destructOwnedHeap();
      MemberContainer<DataType> _shareds;
      std::vector<Phase*> _sharedPhases;
      std::vector<TriggeredFunction*> _sharedTriggeredFunctions;
      MemberContainer<DataType> _optionalSharedServices;
};


#endif // SharedCCBase_H
