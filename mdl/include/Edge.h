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

#ifndef Edge_H
#define Edge_H
#include "Mdl.h"

#include "SharedCCBase.h"
#include "EdgeConnection.h"
#include <memory>

class Generatable;
class Connection;

class Edge : public SharedCCBase {

   public:
      Edge(const std::string& fileName);
      Edge(const Edge& rv);
      Edge operator=(const Edge& rv);
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const;
      virtual ~Edge();
      virtual std::string getType() const;
      virtual std::string generateExtra() const;
      EdgeConnection* getPreNode();
      void setPreNode(std::unique_ptr<EdgeConnection>&& con);
      EdgeConnection* getPostNode();
      void setPostNode(std::unique_ptr<EdgeConnection>&& con);

   protected:
      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
      virtual void addExtraInstanceMethods(Class& instance) const;
      virtual void addExtraInstanceProxyMethods(Class& instance) const;
      virtual void addExtraCompCategoryBaseMethods(Class& instance) const;
      virtual void addExtraCompCategoryMethods(Class& instance) const;

      std::string getEdgeConnectionCommonBody(
	 EdgeConnection& connection) const;
      virtual void addCompCategoryBaseConstructorMethod(Class& instance) const;

   private:
      void copyOwnedHeap(const Edge& rv);
      void destructOwnedHeap();
      EdgeConnection* _preNode;
      EdgeConnection* _postNode;      
};


#endif // Edge_H
