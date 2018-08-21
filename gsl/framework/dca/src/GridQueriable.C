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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "GridQueriable.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "NodeQueriable.h"
#include "Repertoire.h"
#include "GridDataItem.h"
#include "NodeAccessor.h"

#include <sstream>
#include <iostream>

GridQueriable::GridQueriable(Grid* grid)
{
   _densityIdxIdx = -1;          // initialize to -1 for sake of safe copy construction only
   _layerIdx = -1;

   _grid = grid;
   _size = _grid->getSize();
   _publisherQueriable = false;
   std::ostringstream ostr;
   QueryField* ptrQF;
   _queriableName = _grid->getName();
   _queriableDescription = "Access nodes in a grid by grid coordinates, index, and layer:";
   _queriableType = "Grid";
   for (unsigned i = 0; i<_size.size(); i++) {
      std::auto_ptr<QueryField> aptrQF(new QueryField(QueryField::VALUE));
      ptrQF = aptrQF.get();
      ostr<<"Coord"<<i;
      ptrQF->setName(ostr.str());
      ostr.str("");
      ptrQF->setDescription("Grid Coordinate.");
      ostr<<"[0.."<<_size[i]-1<<"]";
      ptrQF->setFormat(ostr.str());
      ostr.str("");
      _queryDescriptor.addQueryField(aptrQF);
   }

   std::auto_ptr<QueryField> aptr_densityQF(new QueryField(QueryField::VALUE));
   aptr_densityQF->setName("Index");
   aptr_densityQF->setDescription("Index of node within Grid Coordinate.");
   ostr.str("");
   ostr<<"[0.."<<_grid->getMaxDensity()-1<<"]";
   aptr_densityQF->setFormat(ostr.str());
   _densityIdxIdx = _queryDescriptor.addQueryField(aptr_densityQF);

   std::auto_ptr<QueryField> aptr_layerQF(new QueryField(QueryField::ENUM));
   aptr_layerQF->setName("Grid Layer");
   aptr_layerQF->setDescription("Enumerated list of Grid Layers.");
   for (unsigned i = 0; i<_grid->size(); i++) {
      std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(((*_grid)[i])->getName(), ((*_grid)[i])->getModelName()));
      aptr_layerQF->addEnumEntry(aptrEnumEntry);
   }
   _layerIdx = _queryDescriptor.addQueryField(aptr_layerQF);
}


GridQueriable::GridQueriable(const GridQueriable & q)
   : Queriable(q), _size(q._size), _grid(q._grid), _layerIdx(q._layerIdx),
     _densityIdxIdx(q._densityIdxIdx)
{
}


void GridQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   GridDataItem* di = new GridDataItem;
   di->setGrid(_grid);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> GridQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::vector<QueryField*> & queryFields = _queryDescriptor.getQueryFields();
   std::auto_ptr<QueryResult> qr(new QueryResult());
   std::vector<int> coords;
   int didx;
   int idx;
   bool inRange = true;
   // get coordinates from fields
   for (unsigned i = 0; i < _size.size(); i++) {
      std::istringstream istr(queryFields[i]->getField());
      istr>>idx;
      if ((idx<0) || (idx>_size[i])) inRange = false;
      coords.push_back(idx);
   }
   if (inRange) {
      // get density index from field
      std::istringstream istr(queryFields[_densityIdxIdx]->getField());
      istr>>didx;
      if (didx<0) inRange = false;
   }
   if (inRange) {
      // get layer from field
      Grid::iterator end = _grid->end();
      for (Grid::iterator iter = _grid->begin(); iter != end; iter++) {
         if (queryFields[_layerIdx]->getField() == (*iter)->getName()) {
            std::ostringstream ostr;
            ostr<<"[0.."<<((*iter)->getMaxDensity())-1<<"]";
            (_queryDescriptor.getQueryFields())[_densityIdxIdx]->setFormat(ostr.str());
            // check if didx is valid for grid layer
            if (didx<(*iter)->getDensity(coords)) {
               qr->_numFound++;
               if ((qr->_numFound>maxItem) || (qr->_numFound-minItem>searchSize)) {
                  qr->_searchCompleted=false;
                  break;
               }
               else if (qr->_numFound>minItem) {
                  NodeDescriptor* nd = (*iter)->getNodeAccessor()->getNodeDescriptor(coords, didx);
                  std::auto_ptr<Queriable> aptrQueriable(new NodeQueriable(nd));
                  qr->addQueriable(aptrQueriable);
               }
            }
         }
      }
      qr->_numFound-=minItem;
   }
   return qr;
}


Publisher* GridQueriable::getQPublisher()
{
   return 0;
}


void GridQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new GridQueriable(*this));
}


GridQueriable::~GridQueriable()
{
}
