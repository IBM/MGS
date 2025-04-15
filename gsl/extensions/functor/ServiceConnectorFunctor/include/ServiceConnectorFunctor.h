// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ServiceConnectorFunctor_H
#define ServiceConnectorFunctor_H
#include "Lens.h"

#include "CG_ServiceConnectorFunctorBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>
#include <string>

class Constant;
class Variable;
class NodeSet;
class EdgeSet;
class Publishable;
class ServiceAcceptor;
class ConstantDataItem;
class VariableDataItem;
class EdgeSetDataItem;
class NodeSetDataItem;
class Service;

class ServiceConnectorFunctor : public CG_ServiceConnectorFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end);
      ServiceConnectorFunctor();
      virtual ~ServiceConnectorFunctor();
      virtual void duplicate(std::unique_ptr<ServiceConnectorFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ServiceConnectorFunctorBase>&& dup) const;   
   private:
      inline void connectToService(Service* service, 
				   const std::string& acceptorName) const;
      inline void connectToService(Service* service, Variable* elem, 
				   const std::string& acceptorName) const;
      inline void connectToService(Service* service, NodeSet* elem, 
				   const std::string& acceptorName) const;
      inline void connectToService(Service* service, EdgeSet* elem, 
				   const std::string& acceptorName) const;

      class ServiceConnectorElement 
      {
	 public:
	    ServiceConnectorElement(const std::string& serviceName, 
				    const std::string& acceptorName);
	    ServiceConnectorElement(std::unique_ptr<Functor>& functor,
				    const std::string& acceptorName);
	    ~ServiceConnectorElement();
	    ServiceConnectorElement(const ServiceConnectorElement& rv);
	    ServiceConnectorElement& operator=(
	       const ServiceConnectorElement& rv);
	    
	    inline void establishConnection(ServiceConnectorFunctor& parent);

	 private:
	    void copyOwnedHeap(const ServiceConnectorElement& rv);
	    void destructOwnedHeap();

	    void stringConnection(ServiceConnectorFunctor& parent);
	    void functorConnection(ServiceConnectorFunctor& parent);
	    
	    std::string _serviceName;
	    std::string _acceptorName;
	    Functor* _functor;
	    
      };

      enum COMPONENTS {_Constant, _Variable, _NodeSet, _EdgeSet};
      COMPONENTS _sourceType, _destinationType;
      ConstantDataItem* _sourceConstantDI;
      VariableDataItem* _sourceVariableDI;
      VariableDataItem* _destinationVariableDI;
      EdgeSetDataItem* _sourceEdgeSetDI;
      EdgeSetDataItem* _destinationEdgeSetDI;
      NodeSetDataItem* _sourceNodeSetDI;
      NodeSetDataItem* _destinationNodeSetDI;
      LensContext* _execContext;

      std::vector<ServiceConnectorElement> _elements;

};

#endif
