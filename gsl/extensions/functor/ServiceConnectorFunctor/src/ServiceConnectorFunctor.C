// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "ServiceConnectorFunctor.h"
#include "CG_ServiceConnectorFunctorBase.h"
#include "LensContext.h"
#include <memory>
#include <sstream>
#include <cassert>

#include "Constant.h"
#include "Variable.h"
#include "VariableInstanceAccessor.h"
#include "VariableCompCategoryBase.h"
#include "NodeSet.h"
#include "EdgeSet.h"
#include "Publishable.h"
#include "ServiceAcceptor.h"
#include "Service.h"
#include "ConstantDataItem.h"
#include "VariableDataItem.h"
#include "NodeSetDataItem.h"
#include "EdgeSetDataItem.h"
#include "NodeDataItem.h"
#include "EdgeDataItem.h"
#include "CustomStringDataItem.h"
#include "FunctorDataItem.h"
#include "ServiceDataItem.h"
#include "Publisher.h"
#include "Node.h"
#include "Edge.h"

//#include "CG_LifeNode.h"

void ServiceConnectorFunctor::userInitialize(LensContext* CG_c) {}

void ServiceConnectorFunctor::userExecute(
    LensContext* CG_c, std::vector<DataItem*>::const_iterator begin,
    std::vector<DataItem*>::const_iterator end)
{
#ifdef HAVE_MPI
//   int mySpaceId;
//   MPI_Comm_rank(MPI_COMM_WORLD, &mySpaceId);
#endif

  _elements.clear();
  _execContext = CG_c;
  std::string mes = "";
  mes =
      mes + "ServiceConnectorFunctor operates on four arguments:\n" +
      "The last two arguments can repeat for multiple service connections.\n" +
      "1) source: [Constant | Variable | NodeSet | EdgeSet]\n" +
      "2) destination: [Variable | NodeSet | EdgeSet]\n" +
      "3) Service name or functor  from source [String literal | functor]\n" +
      "4) Acceptor name from destination [String literal]";
  int size = end - begin;
  if ((size < 4) || (size % 2))
  {  // (size % 2) == -size is odd-
    throw SyntaxErrorException(mes);
  }

  std::vector<DataItem*>::const_iterator it = begin;

  _sourceConstantDI = dynamic_cast<ConstantDataItem*>(*it);
  if (_sourceConstantDI)
  {
    _sourceType = _Constant;
  }
  else
  {
    _sourceVariableDI = dynamic_cast<VariableDataItem*>(*it);
    if (_sourceVariableDI)
    {
      _sourceType = _Variable;
    }
    else
    {
      _sourceNodeSetDI = dynamic_cast<NodeSetDataItem*>(*it);
      if (_sourceNodeSetDI)
      {
        _sourceType = _NodeSet;
      }
      else
      {
        _sourceEdgeSetDI = dynamic_cast<EdgeSetDataItem*>(*it);
        if (_sourceEdgeSetDI)
        {
          _sourceType = _EdgeSet;
        }
        else
        {
          throw SyntaxErrorException("Argument 1 is non-compliant.\n" + mes);
        }
      }
    }
  }

  ++it;

  _destinationVariableDI = dynamic_cast<VariableDataItem*>(*it);
  if (_destinationVariableDI)
  {
    _destinationType = _Variable;
  }
  else
  {
    _destinationNodeSetDI = dynamic_cast<NodeSetDataItem*>(*it);
    if (_destinationNodeSetDI)
    {
      _destinationType = _NodeSet;
    }
    else
    {
      _destinationEdgeSetDI = dynamic_cast<EdgeSetDataItem*>(*it);
      if (_destinationEdgeSetDI)
      {
        _destinationType = _EdgeSet;
      }
      else
      {
        throw SyntaxErrorException("Argument 2 is non-compliant.\n" + mes);
      }
    }
  }

  ++it;

  CustomStringDataItem* stringDI;
  FunctorDataItem* functorDI;
  std::string serviceName;
  std::unique_ptr<Functor> functorAp;
  while (it != end)
  {
    serviceName = "";
    stringDI = dynamic_cast<CustomStringDataItem*>(*it);
    if (stringDI == 0)
    {
      functorDI = dynamic_cast<FunctorDataItem*>(*it);

      if (functorDI == 0)
      {
        std::ostringstream os;
        os << "Argument " << (it - begin) << " is non-compliant.\n" << mes;
        throw SyntaxErrorException(os.str());
      }
      else
      {
        functorDI->getFunctor()->duplicate(std::move(functorAp));
      }
    }
    else
    {
      serviceName = stringDI->getString();  //
    }

    ++it;

    stringDI = dynamic_cast<CustomStringDataItem*>(*it);
    if (stringDI == 0)
    {
      std::ostringstream os;
      os << "Argument " << (it - begin) << " is non-compliant.\n" << mes;
      throw SyntaxErrorException(os.str());
    }
    if (serviceName == "")
    {
      _elements.push_back(
          ServiceConnectorElement(functorAp, stringDI->getString()));
    }
    else
    {
      _elements.push_back(
          ServiceConnectorElement(serviceName, stringDI->getString()));
    }

    ++it;
  }

  std::vector<ServiceConnectorElement>::iterator elemIt,
      elemEnd = _elements.end();
  for (elemIt = _elements.begin(); elemIt != elemEnd; ++elemIt)
  {
    elemIt->establishConnection(*this);
  }
}

ServiceConnectorFunctor::ServiceConnectorFunctor()
    : CG_ServiceConnectorFunctorBase(),
      _sourceConstantDI(0),
      _sourceVariableDI(0),
      _destinationVariableDI(0),
      _sourceEdgeSetDI(0),
      _destinationEdgeSetDI(0),
      _sourceNodeSetDI(0),
      _destinationNodeSetDI(0),
      _execContext(0)
{
}

ServiceConnectorFunctor::~ServiceConnectorFunctor() {}

void ServiceConnectorFunctor::duplicate(
    std::unique_ptr<ServiceConnectorFunctor>&& dup) const
{
  dup.reset(new ServiceConnectorFunctor(*this));
}

void ServiceConnectorFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
  dup.reset(new ServiceConnectorFunctor(*this));
}

void ServiceConnectorFunctor::duplicate(
    std::unique_ptr<CG_ServiceConnectorFunctorBase>&& dup) const
{
  dup.reset(new ServiceConnectorFunctor(*this));
}

void ServiceConnectorFunctor::connectToService(
    Service* service, const std::string& acceptorName) const
{
#ifdef HAVE_MPI
//   int mySpaceId;
//   MPI_Comm_rank(MPI_COMM_WORLD, &mySpaceId);
#endif

  if (_destinationType == _Variable)
  {
    // need to determine both ends' partitionID and decide which whether need to
    // create
    // an proxy. It is always that create an proxy for the pre-node or
    // pre-variable.
    // similiar to the case in LensConnector: if both ends are not in current
    // memory
    // space we do nothing.
    Variable* av = _destinationVariableDI->getVariable()->getVariable();
    if (av)
    {
      connectToService(
          //	    service, _destinationVariableDI->getVariable(),
          //acceptorName);
          service, _destinationVariableDI->getVariable()->getVariable(),
          acceptorName);  // added by Jizhu Lu on 03/16/2006
    }
  }
  else if (_destinationType == _NodeSet)
  {
    connectToService(service, _destinationNodeSetDI->getNodeSet(),
                     acceptorName);
  }
  else if (_destinationType == _EdgeSet)
  {
    connectToService(service, _destinationEdgeSetDI->getEdgeSet(),
                     acceptorName);
  }
  else
  {
    assert(0);
  }
}

void ServiceConnectorFunctor::connectToService(
    Service* service, Variable* elem, const std::string& acceptorName) const
{
  assert(elem != NULL);
  elem->acceptService(service, acceptorName);
}

void ServiceConnectorFunctor::connectToService(
    Service* service, NodeSet* elem, const std::string& acceptorName) const
{
  std::vector<NodeDescriptor*> nodes;
  elem->getNodes(nodes);
  std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end = nodes.end();
  for (; it != end; ++it)
  {
    // @TODO Distributed local filter
    //      if ((*it)->getNode() && (*it)->getNode()->hasService())     //
    //      modified by Jizhu Lu on 01/13/2006   // commented out by Jizhu Lu on
    //      06/14/2006
    if ((*it)->getNode())  // modified by Jizhu Lu on 06/14/2006
      (*it)->getNode()->acceptService(service, acceptorName);
  }
}

void ServiceConnectorFunctor::connectToService(
    Service* service, EdgeSet* elem, const std::string& acceptorName) const
{
  std::vector<Edge*>& edges = elem->getEdges();
  std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
  for (; it != end; ++it)
  {
    (*it)->acceptService(service, acceptorName);
  }
}

// ----------------- Inner class methods -----------------

ServiceConnectorFunctor::ServiceConnectorElement::ServiceConnectorElement(
    const std::string& serviceName, const std::string& acceptorName)
    : _serviceName(serviceName), _acceptorName(acceptorName), _functor(0)
{
}

ServiceConnectorFunctor::ServiceConnectorElement::ServiceConnectorElement(
    std::unique_ptr<Functor>& functor, const std::string& acceptorName)
    : _serviceName(""), _acceptorName(acceptorName), _functor(0)
{
  _functor = functor.release();
}

ServiceConnectorFunctor::ServiceConnectorElement::~ServiceConnectorElement()
{
  destructOwnedHeap();
}

ServiceConnectorFunctor::ServiceConnectorElement::ServiceConnectorElement(
    const ServiceConnectorFunctor::ServiceConnectorElement& rv)
    : _serviceName(rv._serviceName),
      _acceptorName(rv._acceptorName),
      _functor(0)
{
  copyOwnedHeap(rv);
}

ServiceConnectorFunctor::ServiceConnectorElement&
    ServiceConnectorFunctor::ServiceConnectorElement::
        operator=(const ServiceConnectorFunctor::ServiceConnectorElement& rv)
{
  if (this != &rv)
  {
    destructOwnedHeap();
    copyOwnedHeap(rv);
    _serviceName = rv._serviceName;
    _acceptorName = rv._acceptorName;
  }
  return *this;
}

void ServiceConnectorFunctor::ServiceConnectorElement::establishConnection(
    ServiceConnectorFunctor& parent)
{
  if (_serviceName == "")
  {
    functorConnection(parent);
  }
  else
  {
    stringConnection(parent);
  }
}

void ServiceConnectorFunctor::ServiceConnectorElement::stringConnection(
    ServiceConnectorFunctor& parent)
{
#ifdef HAVE_MPI
  int mySpaceId;
  MPI_Comm_rank(MPI_COMM_WORLD, &mySpaceId);
#endif
  Service* service = 0;
  if (parent._sourceType == _Constant)
  {
    service =
        parent._sourceConstantDI->getConstant()->getPublisher()->getService(
            _serviceName);
    parent.connectToService(service, _acceptorName);
  }
  else if (parent._sourceType == _Variable)
  {
    // create an variable proxy if getVariable() returns a NULL
    VariableInstanceAccessor* via = parent._sourceVariableDI->getVariable();
    unsigned variableIndex = via->getVariableIndex();
    Variable* av = via->getVariable();
    if (!av)
    {
#ifdef HAVE_MPI
      int fromPartitionId =
          parent._execContext->sim->getGranule(variableIndex)->getPartitionId();
      if (fromPartitionId != mySpaceId)
      {
        via->getVariableType()->allocateProxy(fromPartitionId, av);
        av->setVariableIndex(via->getVariableIndex());
        via->setVariable(av);
      }
#else
      av->setVariableIndex(via->getVariableIndex());
      via->setVariable(av);
#endif
    }
    assert(av->getPublisher() != NULL);
    service = av->getPublisher()->getService(_serviceName);
    parent.connectToService(service, _acceptorName);
  }
  else if (parent._sourceType == _NodeSet)
  {
    std::vector<NodeDescriptor*> nodes;
    parent._sourceNodeSetDI->getNodeSet()->getNodes(nodes);
    std::vector<NodeDescriptor*>::iterator it = nodes.begin(),
                                           end = nodes.end();
    for (; it != end; ++it)
    {
      // @TODO Distributed local filter
      //         if ((*it)->getNode() && (*it)->getNode()->hasService()) {
      //         // modified by Jizhu Lu on 01/12/2006  // commented out by
      //         Jizhu Lu on 06/14/2006
      if ((*it)->getNode())
      {  // modified by Jizhu Lu on 06/14/2006
        service = (*it)->getNode()->getPublisher()->getService(_serviceName);
        parent.connectToService(service, _acceptorName);
      }
    }
  }
  else if (parent._sourceType == _EdgeSet)
  {
    std::vector<Edge*>& edges =
        parent._sourceEdgeSetDI->getEdgeSet()->getEdges();
    std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
    for (; it != end; ++it)
    {
      service = (*it)->getPublisher()->getService(_serviceName);
      parent.connectToService(service, _acceptorName);
    }
  }
  else
  {
    assert(0);
  }
}

void ServiceConnectorFunctor::ServiceConnectorElement::functorConnection(
    ServiceConnectorFunctor& parent)
{
  std::unique_ptr<DataItem> retValue;
  std::vector<DataItem*> args;
  // If the source is constant or variable, than new dataitems that
  // doesn't own their _data is needed for ConstantDataItem and
  // VariableDataItem. e.g., PConstantDataItem PVariableDataItem.
  assert(parent._sourceType != _Constant);
  assert(parent._sourceType != _Variable);
  if (parent._sourceType == _NodeSet)
  {
    std::vector<NodeDescriptor*> nodes;
    parent._sourceNodeSetDI->getNodeSet()->getNodes(nodes);
    std::vector<NodeDescriptor*>::iterator it = nodes.begin(),
                                           end = nodes.end();
    NodeDataItem* nodeDI = new NodeDataItem();
    ServiceDataItem* serviceDI;
    for (; it != end; ++it)
    {
      // @TODO Distributed local filter
      //         if ((*it)->getNode() && (*it)->getNode()->hasService()) {  //
      //         modified by Jizhu Lu on 01/13/2006   // commented out by Jizhu
      //         Lu on 06/14/2006
      if ((*it)->getNode())
      {  // modified by Jizhu Lu on 06/14/2006
         //            std::cerr <<
         //            "ServiceConnectorElement::functorConnection(),in loop,
         //            value=" <<
         //            dynamic_cast<CG_LifeNode>((*it)->getNode())->CG_get_ValueProducer_value()
         //            << std::endl;
        nodeDI->setNode((*it)->getNode());
        args.clear();
        args.push_back(nodeDI);
        _functor->execute(parent._execContext, args, retValue);
        serviceDI = dynamic_cast<ServiceDataItem*>(retValue.get());
        if (serviceDI == 0)
        {
          throw SyntaxErrorException(
              "Functor did not return a Service from a node in "
              "ServiceConnector.\n");
        }
        parent.connectToService(serviceDI->getService(), _acceptorName);
      }
    }
    delete nodeDI;
  }
  else if (parent._sourceType == _EdgeSet)
  {
    std::vector<Edge*>& edges =
        parent._sourceEdgeSetDI->getEdgeSet()->getEdges();
    std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
    EdgeDataItem* edgeDI = new EdgeDataItem();
    ServiceDataItem* serviceDI;
    for (; it != end; ++it)
    {
      edgeDI->setEdge(*it);
      args.clear();
      args.push_back(edgeDI);
      _functor->execute(parent._execContext, args, retValue);
      serviceDI = dynamic_cast<ServiceDataItem*>(retValue.get());
      if (serviceDI == 0)
      {
        throw SyntaxErrorException(
            "Functor did not return a Service from an edge in "
            "ServiceConnector.\n");
      }
      parent.connectToService(serviceDI->getService(), _acceptorName);
    }
  }
  else
  {
    assert(0);
  }
}

void ServiceConnectorFunctor::ServiceConnectorElement::copyOwnedHeap(
    const ServiceConnectorFunctor::ServiceConnectorElement& rv)
{
  if (rv._functor)
  {
    std::unique_ptr<Functor> dup;
    rv._functor->duplicate(std::move(dup));
    _functor = dup.release();
  }
  else
  {
    _functor = 0;
  }
}

void ServiceConnectorFunctor::ServiceConnectorElement::destructOwnedHeap()
{
  delete _functor;
}
