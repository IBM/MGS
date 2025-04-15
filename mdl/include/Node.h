// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Node_H
#define Node_H
#include "Mdl.h"

#include "SharedCCBase.h"

#include <memory>

class Generatable;

class Node : public SharedCCBase {
   public:
      Node(const std::string& fileName);      
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const;
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
   private:
      void _add_allocateNode_Method(Class& instance) const;
      void _add_allocateNodes_Method(Class& instance) const;
      void _add_allocateProxies_Method(Class& instance) const;
};


#endif // Node_H
