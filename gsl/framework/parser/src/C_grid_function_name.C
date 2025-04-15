// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_grid_function_name.h"
#include "C_declarator.h"
#include "FunctorDataItem.h"
#include "LensContext.h"
#include "NDPairListDataItem.h"
#include "NDPairList.h"
#include "IntArrayDataItem.h"
#include "LayoutFunctor.h"
#include "NodeInitializerFunctor.h"
#include "Functor.h"
#include "CustomStringDataItem.h"
#include "NodeTypeDataItem.h"
#include "NodeSetDataItem.h"
#include "GranuleMapperDataItem.h"
#include "IntArrayDataItem.h"
#include "NodeSet.h"
#include "C_argument_list.h"
#include "LayerDefinitionContext.h"
#include "NodeType.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production_grid.h"
#include "Simulation.h"
#include "GranuleMapper.h"
#include <stdio.h>
#include <string>

void C_grid_function_name::internalExecute(LensContext* c, Grid* grid) {
  if (_declarator) _declarator->execute(c);
  _argList->execute(c);
  if (_type == _LAYER)
    layers(c, grid);
  else if (_type == _INITNODES)
    initNodes(c, grid);
}

C_grid_function_name::C_grid_function_name(const C_grid_function_name& rv)
    : C_production_grid(rv), _type(rv._type), _argList(0), _declarator(0) {
  if (rv._argList) {
    _argList = rv._argList->duplicate();
  }
  if (rv._declarator) {
    _declarator = rv._declarator->duplicate();
  }
}

C_grid_function_name::C_grid_function_name(C_declarator* d, C_argument_list* a,
                                           SyntaxError* error)
    : C_production_grid(error), _type(_LAYER), _argList(a), _declarator(d) {}

C_grid_function_name::C_grid_function_name(C_argument_list* a,
                                           SyntaxError* error)
    : C_production_grid(error),
      _type(_INITNODES),
      _argList(a),
      _declarator(0) {}

C_grid_function_name* C_grid_function_name::duplicate() const {
  return new C_grid_function_name(*this);
}

C_grid_function_name::~C_grid_function_name() {
  delete _declarator;
  delete _argList;
}

void C_grid_function_name::initNodes(LensContext* c, Grid* grid) {
  const std::vector<DataItem*>* args = _argList->getVectorDataItem();
  if (args->size() < 2) {
    std::string mes =
        "Expected at least two arguments to 'initnodes' in grid definition.";
    throwError(mes);
  }
  std::vector<DataItem*>::const_iterator a_iter = args->begin();

  // now get the NodeSet
  const DataItem* dataItem = *a_iter++;
  const NodeSetDataItem* nsdi = dynamic_cast<const NodeSetDataItem*>(dataItem);
  if (nsdi == 0) {
    std::string mes = "First argument of 'initnodes' is not a nodeset.";
    throwError(mes);
  }
  NodeSet* nodeset = nsdi->getNodeSet();
  c->layerContext = new LayerDefinitionContext;
  c->layerContext->nodeset = nodeset;
  c->layerContext->grid = nodeset->getGrid();
  for (unsigned int it = 1; it < args->size(); ++it) {
    // now get the NodeInit functors
    dataItem = *a_iter++;
    const FunctorDataItem* fdi = dynamic_cast<const FunctorDataItem*>(dataItem);
    if (fdi == 0) {
      std::string mes = "Subsequent argument of 'initnodes' is not a functor.";
      throwError(mes);
    }
    Functor* functor = fdi->getFunctor();
    const std::string& category = functor->getCategory();
    if (category != NodeInitializerFunctor::_category) {
      std::string mes =
          "Subsequent argument of 'initnodes' is not a NodeInitializer "
          "functor, instead it is: ";
      mes += functor->getCategory();
      throwError(mes);
    }

    // call the functor to initialize the nodes
    std::unique_ptr<DataItem> returnValue;
    std::vector<DataItem*> emptyArgs;
    functor->execute(c, emptyArgs, returnValue);
  }

  // clean up
  delete c->layerContext;
  c->layerContext = 0;
}

/* being called when a 'Layers'
 * statement is detected in GSL
 */
void C_grid_function_name::layers(LensContext* c, Grid* grid) {
  std::string name = _declarator->getName();

  const std::vector<DataItem*>* args = _argList->getVectorDataItem();
  if (args->size() != 3 && args->size() != 4) {
    std::string mes =
        "Expected four or five arguments to 'layer' in grid definition.";
    throwError(mes);
  }
  std::vector<DataItem*>::const_iterator a_iter = args->begin();

  // now get the NodeType
  const DataItem* dataItem = *a_iter++;
  const NodeTypeDataItem* ntdi =
      dynamic_cast<const NodeTypeDataItem*>(dataItem);
  if (ntdi == 0) {
    std::string mes = "Second argument of 'layer' is not a node type.";
    throwError(mes);
  }
  NodeType* model = ntdi->getNodeType();

  // now get the Layout functor
  dataItem = *a_iter++;
  const FunctorDataItem* fdi = dynamic_cast<const FunctorDataItem*>(dataItem);
  if (fdi == 0) {
    std::string mes = "Third argument of 'layer' is not a functor.";
    throwError(mes);
  }
  Functor* functor = fdi->getFunctor();
  const std::string& category = functor->getCategory();
  if (category != LayoutFunctor::_category) {
    std::string mes =
        "Third argument of 'layer' is not a Layout functor, instead it is: ";
    mes += functor->getCategory();
    throwError(mes);
  }

  // now get the NDPairList
  dataItem = *a_iter++;
  const NDPairListDataItem* ndpdi =
      dynamic_cast<const NDPairListDataItem*>(dataItem);
  if (ndpdi == 0) {
    std::string mes = "Fourth argument of 'layer' is not an NDPairList.";
    throwError(mes);
  }
  NDPairList* ndpairlist = ndpdi->getNDPairList();

  // Get density vector from the layout functor
  std::unique_ptr<DataItem> returnValue;
  std::vector<DataItem*> newArgs;

  c->layerContext = new LayerDefinitionContext;
  c->layerContext->nodeset = 0;
  c->layerContext->grid = grid;

  functor->execute(c, newArgs, returnValue);
  dataItem = returnValue.get();
  const IntArrayDataItem* iadi =
      dynamic_cast<const IntArrayDataItem*>(dataItem);
  if (iadi == 0) {
    std::string mes = "Layout functor did not return density vector.";
    throwError(mes);
  }
  const std::vector<int>* density = iadi->getIntVector();

  GranuleMapper* granuleMapper;

  // now get the GranuleMapper
  if (args->size() == 4) {
    dataItem = *a_iter++;
    const GranuleMapperDataItem* gmdi =
        dynamic_cast<const GranuleMapperDataItem*>(dataItem);
    if (gmdi == 0) {
      std::string mes = "Fifth argument of 'layer' is not a GranuleMapper.";
      throwError(mes);
    }
    granuleMapper = gmdi->getGranuleMapper();
  } else {
    if (c->sim->isGranuleMapperPass() || c->sim->isCostAggregationPass()) {
      std::vector<DataItem*> gmargs;
      CustomStringDataItem descrDI("Default Volume Granule Mapper.");
      gmargs.push_back(&descrDI);

      const std::vector<int>& sz = grid->getSize();
      std::vector<int> coord(1);
      coord[0] = sz.size();
      IntArrayDataItem dimDI(coord);
      coord[0] = 0;
      std::vector<int>::const_reverse_iterator iter2 = sz.rbegin(),
                                               end2 = sz.rend();
      for (; iter2 != end2; ++iter2) {
        dimDI.setInt(coord, (*iter2));
        coord[0]++;
      }
      gmargs.push_back(&dimDI);

      coord[0] = density->size();
      IntArrayDataItem denDI(coord);
      coord[0] = 0;
      std::vector<int>::const_iterator iter = density->begin(),
                                       end = density->end();
      for (; iter != end; ++iter) {
        denDI.setInt(coord, (*iter));
        coord[0]++;
      }
      gmargs.push_back(&denDI);

      GranuleMapperType* gt =
          c->sim->getGranuleMapperType("VolumeGranuleMapper");
      std::unique_ptr<GranuleMapper> apgm;
      gt->getGranuleMapper(gmargs, apgm);
      granuleMapper = apgm.get();

      granuleMapper->setIndex(c->sim->getGranuleMapperCount());
      c->sim->addGranuleMapper(apgm);
    } else {
      granuleMapper = c->sim->getGranuleMapper(c->sim->getGranuleMapperCount());
    }
    c->sim->incrementGranuleMapperCount();
  }

  delete c->layerContext;
  c->layerContext = 0;

  // add the layer
  try {
    grid->addLayer(*density, name, model, *ndpairlist,
                   granuleMapper->getIndex());
  } catch (SyntaxErrorException& e) {
    throwError(e.getError());
  }
}

void C_grid_function_name::checkChildren() {
  if (_argList) {
    _argList->checkChildren();
    if (_argList->isError()) {
      setError();
    }
  }
  if (_declarator) {
    _declarator->checkChildren();
    if (_declarator->isError()) {
      setError();
    }
  }
}

void C_grid_function_name::recursivePrint() {
  if (_argList) {
    _argList->recursivePrint();
  }
  if (_declarator) {
    _declarator->recursivePrint();
  }
  printErrorMessage();
}
