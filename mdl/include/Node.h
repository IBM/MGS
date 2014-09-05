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

#ifndef Node_H
#define Node_H
#include "Mdl.h"

#include "SharedCCBase.h"

#include <memory>

class Generatable;

class Node : public SharedCCBase {

   public:
      Node(const std::string& fileName);      
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual ~Node();
      virtual std::string getType() const;

   protected:
      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();

      void generateNodeAccessor();
      void generateGridLayerData();
      void generateWorkUnitGridLayers();
      void generateResourceFile();

      std::string getNodeAccessorName() const {
	 return PREFIX + getName() + "NodeAccessor";
      }
      std::string getGridLayerDataName() const {
	 return PREFIX + getName() + "GridLayerData";
      }
      std::string getWorkUnitGridLayersName() const {
	 return getWorkUnitCommonName("GridLayers");
      }

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
      virtual void addExtraInstanceMethods(Class& instance) const;
      virtual void addExtraInstanceProxyMethods(Class& instance) const;
      virtual void addExtraCompCategoryBaseMethods(Class& instance) const;
      virtual void addExtraCompCategoryMethods(Class& instance) const;
      virtual void addCompCategoryBaseConstructorMethod(Class& instance) const;
};


#endif // Node_H
