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

#include "C_service.h"
#include "C_query_path_product.h"
#include "C_argument_declarator.h"
#include "C_argument_string.h"
#include "C_declarator.h"
#include "Service.h"
#include "Publisher.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "DataItem.h"
#include "NodeSetDataItem.h"
#include "EdgeSetDataItem.h"
#include "VariableDataItem.h"
#include "NodeSet.h"
#include "EdgeSet.h"
#include "Node.h"
#include "Edge.h"
#include "Variable.h"
#include "VariableInstanceAccessor.h"

#include <cassert>

void C_service::internalExecute(LensContext* c)
{
  if (_queryPathProduct)
  {
    _queryPathProduct->execute(c);
    _service = _queryPathProduct->getService();
  }
  else
  {
    assert(_declarator != 0);
    assert(_string != 0);
    _declarator->execute(c);
    DataItem* di =
        const_cast<DataItem*>(c->symTable.getEntry(_declarator->getName()));
    if (!di)
    {
      std::string mes =
          "Argument " + _declarator->getName() + " was not declared";
      throwError(mes);
    }

    NodeSetDataItem* nsdi = dynamic_cast<NodeSetDataItem*>(di);
    if (nsdi)
    {
      std::vector<NodeDescriptor*> nodes;
      nsdi->getNodeSet()->getNodes(nodes);
      // @TODO Distributed local filter
      if (nodes[0]->getNode())  // added by Jizhu Lu on 12/04/2005
        _service = nodes[0]->getNode()->getPublisher()->getService(*_string);
    }
    else
    {
      EdgeSetDataItem* esdi = dynamic_cast<EdgeSetDataItem*>(di);
      if (esdi)
      {
        std::vector<Edge*>& edges = esdi->getEdgeSet()->getEdges();
        _service = edges[0]->getPublisher()->getService(*_string);
      }
      else
      {
        VariableDataItem* vdi = dynamic_cast<VariableDataItem*>(di);
        if (vdi)
        {
          //	       _service =
          //vdi->getVariable()->getPublisher()->getService(
          _service = vdi->getVariable(0)
                         ->getVariable()
                         ->getPublisher()
                         ->getService(  // added by Jizhu Lu on 03/16/2006
                             *_string);
        }
        else
        {
          assert(0);
        }
      }
    }
  }
}

C_service::C_service(C_query_path_product* qpp, SyntaxError* error)
    : C_production(error),
      _queryPathProduct(qpp),
      _declarator(0),
      _string(0),
      _service(0)
{
}

C_service::C_service(C_declarator* ad, std::string* as, SyntaxError* error)
    : C_production(error),
      _queryPathProduct(0),
      _declarator(ad),
      _string(as),
      _service(0)
{
}

C_service::C_service(const C_service& rv)
    : C_production(rv),
      _queryPathProduct(0),
      _declarator(0),
      _string(0),
      _service(rv._service)
{
  if (rv._queryPathProduct)
  {
    _queryPathProduct = rv._queryPathProduct->duplicate();
  }
  if (rv._declarator)
  {
    _declarator = rv._declarator->duplicate();
  }
  if (rv._string)
  {
    _string = new std::string(*(rv._string));
  }
}

C_service* C_service::duplicate() const { return new C_service(*this); }

C_service::~C_service()
{
  delete _queryPathProduct;
  delete _declarator;
  delete _string;
}

void C_service::checkChildren()
{
  if (_queryPathProduct)
  {
    _queryPathProduct->checkChildren();
    if (_queryPathProduct->isError())
    {
      setError();
    }
  }
  if (_declarator)
  {
    _declarator->checkChildren();
    if (_declarator->isError())
    {
      setError();
    }
  }
}

void C_service::recursivePrint()
{
  if (_queryPathProduct)
  {
    _queryPathProduct->recursivePrint();
  }
  if (_declarator)
  {
    _declarator->recursivePrint();
  }
  printErrorMessage();
}
