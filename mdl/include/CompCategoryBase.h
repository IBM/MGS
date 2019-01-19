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

#ifndef CompCategoryBase_H
#define CompCategoryBase_H
#include "Mdl.h"

#include "InterfaceImplementorBase.h"
#include "DataType.h"
#include "Phase.h"
#include "TriggeredFunction.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <sstream>
#include <set>

class StructType;
class ConnectionIncrement;

class CompCategoryBase : public InterfaceImplementorBase {
   public:
      CompCategoryBase(const std::string& fileName);
      CompCategoryBase(const CompCategoryBase& rv);
      CompCategoryBase& operator=(const CompCategoryBase& rv);
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const =0;
      virtual std::string generateExtra() const;
      virtual std::string getType() const =0;
      virtual ~CompCategoryBase();
      void releaseInAttrPSet(std::auto_ptr<StructType>& iap);
      void setInAttrPSet(std::auto_ptr<StructType>& iap);
      void setTriggeredFunctions(
	 std::auto_ptr<std::vector<TriggeredFunction*> >& triggeredfunction);

      StructType* getInAttrPSet() {
	 return _inAttrPSet;
      }
      virtual void addSharedDataToKernelArgs(Class* c) {};

   protected:
      void generateInAttrPSet();
      void generatePSet();
      void generatePSet(bool use_classType, std::pair<Class::PrimeType, Class::SubType> classType);
      void generateWorkUnitInstance();
      void generateInstance();
      void generateInstance(bool use_classType, std::pair<Class::PrimeType, Class::SubType> classType, Class* ptr=nullptr);
      void generateCompCategoryBase(Class* ptr=nullptr);
      void generateCompCategory();
      void generateTriggerableCallerInstance();
      void generateResourceFile();                // added by Jizhu Lu on 02/09/2006

      std::string getPSetName() const {
	 return getCommonPSetName("");
      }
      std::string getWorkUnitCommonName(const std::string& type) const {
	 return PREFIX + getName() + "WorkUnit" + type;
      }
      std::string getTriggerableCallerCommonName(
	 const std::string& modelName) const {
	 return modelName + "TriggerableCaller";
      }
      std::string getWorkUnitInstanceName() const {
	 return getWorkUnitCommonName("Instance");
      }
      std::string getInstanceName() const {
	 return getName();
      }
      virtual std::string getCompCategoryBaseName() const {
	 return PREFIX + getName() + COMPCATEGORY;
      }
      std::string getCompCategoryName() const {
	 return getName() + COMPCATEGORY;
      }
      std::string getTriggerableCallerInstanceName() const {
	 return getTriggerableCallerCommonName(getInstanceBaseName());
      }

      std::string getFrameworkCompCategoryName() const {
	 return getType() + COMPCATEGORYBASE;
      }

      // Don't call directly, called internally by other generateWorkUnitX
      void generateWorkUnitCommon(const std::string& workUnitType, 
				  const std::string& argumentType, 
				  const std::string& compCatName);
      
      // Don't call directly, 
      // called internally by other generateTriggerableCallerX
      void generateTriggerableCallerCommon(const std::string& modelType);
      
      virtual void addExtraInstanceBaseMethods(Class& instance) const;


      virtual void addExtraInstanceProxyMethods(Class& instance) const;

      // Will be implemented in derived classes.
      virtual void addExtraInstanceMethods(Class& instance) const {
	 return;
      }

      // Will be implemented in derived classes.
      virtual void addExtraCompCategoryBaseMethods(Class& instance) const {
	 return;
      }

      // Will be implemented in derived classes.
      virtual void addExtraCompCategoryMethods(Class& instance) const {
	 return;
      }

      // Used by SharedCCBase
      void isDuplicatePhase(const Phase* phase);

      // Used by SharedCCBase
      void isDuplicateTriggeredFunction(const std::string& name);

      // Creates the function body for getWorkUnits method.
      virtual std::string createGetWorkUnitsMethodBody(
	 const std::string& phaseName, const std::string& workUnits) const;

      /* added by Jizhu Lu on 12/03/2005 */
      virtual std::string createAddNodeMethodBody(std::string, std::string) const;
      virtual std::string createAllocateProxyMethodBody(std::string, std::string) const;
      /**** end of addition -- Jizhu Lu ****/

      std::string createGetWorkUnitsMethodCommonBody(
	 const std::string& phaseName, const std::string& workUnits, 
	 const std::string& instanceName, 
	 const std::vector<Phase*>& phases) const;      

      virtual std::string getCompCategoryBaseConstructorBody() const;

      virtual void addCompCategoryBaseConstructorMethod(
	 Class& instance) const =0;

      void addTriggeredFunctionMethods(
	 Class& instance, 
	 const std::vector<TriggeredFunction*>& functions, 
	 bool pureVirtual) const;

      void addCreateTriggerableCallerMethod(
	 Class& instance, 
	 const std::vector<TriggeredFunction*>* functions,
	 const std::string& triggerableCallerName) const;

      std::string getAddVariableNamesForPhaseFB() const;
      std::string getSetDistributionTemplatesFB() const;
      std::string getResetSendProcessIdIteratorsFB() const;
      std::string getGetSendNextProcessIdFB() const;
      std::string getAtSendProcessIdEndFB() const;
      std::string getResetReceiveProcessIdIteratorsFB() const;
      std::string getGetReceiveNextProcessIdFB() const;
      std::string getAtReceiveProcessIdEndFB() const;

      std::string getSetMemPatternFB() const;
      std::string getGetIndexedBlockFB() const;
      std::string getGetReceiveBlockCreatorFB() const;
      std::string getSendFB() const;
      std::string getGetDemarshallerFB() const;
      std::string getFindDemarshallerFB() const;

      std::string getTemplateFillerCode() const;
      std::string getFindDemarshallerFillerCode() const;          // added by Jizhu Lu on 04/26/2006
      void addDistributionCodeToCC(Class& instance) const;

   private:
      void copyOwnedHeap(const CompCategoryBase& rv);
      void destructOwnedHeap();
      StructType* _inAttrPSet;
      std::vector<TriggeredFunction*>* _triggeredFunctions;
      ConnectionIncrement* _connectionIncrement;                  //  added by Jizhu Lu on 02/09/2006
};


#endif // CompCategoryBase_H
