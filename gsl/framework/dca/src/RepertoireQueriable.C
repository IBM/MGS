// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "RepertoireQueriable.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Repertoire.h"
#include "GridLayerDescriptor.h"
#include "ConnectionSet.h"
#include "GridQueriable.h"
#include "ConnectionSetQueriable.h"
#include "RepertoireDataItem.h"

#include <sstream>
#include <iostream>

RepertoireQueriable::RepertoireQueriable(Repertoire* repertoire)
{
   _gridIdx = -1;                // initialize to -1 for sake of safe copy construction only
   _subRepertoireIdx = -1;
   _connectionSetPreIdx = -1;
   _connectionSetPostIdx = -1;

   _repertoire = repertoire;
   _publisherQueriable = false;
   _queriableName = _repertoire->getName();
   _queriableType = "Repertoire";

   bool gridRep = _repertoire->isGridRepertoire();
   _connectionSetMap = _repertoire->getConnectionSetMap();

   std::ostringstream description;
   description<<"Access";
   if (gridRep) description<<" grid";
   else description<<" subrepertoires";
   if (_connectionSetMap.size()>0) description<<" or connection sets";
   description<<".";
   _queriableDescription = description.str();

   if (gridRep) {
      _queriableList.push_back(new GridQueriable(_repertoire->getGrid()));

      std::auto_ptr<QueryField> aptr_gridQF(new QueryField(QueryField::ENUM));
      aptr_gridQF->setName("Grid");
      aptr_gridQF->setDescription("Provide the queriable Grid of this Grid Repertoire.");
      aptr_gridQF->setFormat("");
      std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry("Grid", "Provide queriable Grid."));
      aptr_gridQF->addEnumEntry(emptyEnum());
      aptr_gridQF->addEnumEntry(aptrEnumEntry);
      _gridIdx = _queryDescriptor.addQueryField(aptr_gridQF);
   }
   else {
      // Make Sub-repertoire Queriable from parent Repertoire
      std::auto_ptr<QueryField> aptr_subRepertoireQF(new QueryField(QueryField::ENUM));
      aptr_subRepertoireQF->setName("Sub-repertoire");
      aptr_subRepertoireQF->setDescription("Enumerated list of Sub-repertoires.");
      aptr_subRepertoireQF->setFormat("");
      // the following empty enum is added to make the gui work
      aptr_subRepertoireQF->addEnumEntry(emptyEnum());
      _subRepertoires = _repertoire->getSubRepertoires();
      std::list<Repertoire*>::const_iterator end = _subRepertoires.end();
      for (std::list<Repertoire*>::const_iterator iter = _subRepertoires.begin(); iter != end; ++iter) {
         _queriableList.push_back(new RepertoireQueriable(*iter));
         std::string descr;
         if ((*iter)->isGridRepertoire()) descr = "Grid Repertoire.";
         else descr = "Composite Repertoire.";
         std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry((*iter)->getName(), descr));
         aptr_subRepertoireQF->addEnumEntry(aptrEnumEntry);
      }
      _subRepertoireIdx = _queryDescriptor.addQueryField(aptr_subRepertoireQF);
   }
   if (_connectionSetMap.size() > 0) {
      std::auto_ptr<QueryField> aptr_connectionSetPreQF(new QueryField(QueryField::ENUM));
      aptr_connectionSetPreQF->setName("Connection Set Pre Identifier");
      aptr_connectionSetPreQF->setDescription("Enumerated list of pre Grid Layer names.");
      aptr_connectionSetPreQF->setFormat("");
      aptr_connectionSetPreQF->addEnumEntry(emptyEnum());
      std::auto_ptr<QueryField>  aptr_connectionSetPostQF(new QueryField(QueryField::ENUM));
      aptr_connectionSetPostQF->setName("Connection Set Post Identifier");
      aptr_connectionSetPostQF->setDescription("Enumerated list of post Grid Layer names.");
      aptr_connectionSetPostQF->setFormat("");
      aptr_connectionSetPostQF->addEnumEntry(emptyEnum());
      std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::const_iterator map_end = _connectionSetMap.end();
      for (std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::const_iterator map_iter = _connectionSetMap.begin();
      map_iter != map_end; ++map_iter) {
         GridLayerDescriptor* prePtr = (*map_iter).first;
         std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(prePtr->getName(), prePtr->getModelName()));
         aptr_connectionSetPreQF->addEnumEntry(aptrEnumEntry);
         std::list<ConnectionSet*> cnxnSetList = (*map_iter).second;
         std::list<ConnectionSet*>::iterator list_end = cnxnSetList.end();
         for (std::list<ConnectionSet*>::iterator list_iter = cnxnSetList.begin(); list_iter != list_end; list_iter++) {
            _queriableList.push_back(new ConnectionSetQueriable(*list_iter));
            bool post_enum_found = false;
            GridLayerDescriptor* postPtr = (*list_iter)->getPostPtr();
            std::vector<EnumEntry*> enumEntries =  aptr_connectionSetPostQF->getEnumEntries();
            std::vector<EnumEntry*>::iterator end = enumEntries.end();
            for (std::vector<EnumEntry*>::iterator enum_iter = enumEntries.begin(); enum_iter!= end; enum_iter++) {
               if ((*enum_iter)->getValue() == postPtr->getName()) post_enum_found = true;
               break;
            }
            if (!post_enum_found) {
               std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(postPtr->getName(), postPtr->getModelName()));
               aptr_connectionSetPostQF->addEnumEntry(aptrEnumEntry);
            }
         }
      }
      _connectionSetPreIdx = _queryDescriptor.addQueryField(aptr_connectionSetPreQF);
      _connectionSetPostIdx = _queryDescriptor.addQueryField(aptr_connectionSetPostQF);
   }
}


RepertoireQueriable::RepertoireQueriable(const RepertoireQueriable & q)
: Queriable(q), _subRepertoires(q._subRepertoires), _connectionSetMap(q._connectionSetMap),
_repertoire(q._repertoire), _gridIdx(q._gridIdx), _subRepertoireIdx(q._subRepertoireIdx),
_connectionSetPreIdx(q._connectionSetPreIdx), _connectionSetPostIdx(q._connectionSetPostIdx)
{
}


void RepertoireQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   RepertoireDataItem* di = new RepertoireDataItem;
   di->setRepertoire(_repertoire);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> RepertoireQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::vector<QueryField*> & queryFields = _queryDescriptor.getQueryFields();
   std::auto_ptr<QueryResult> qr(new QueryResult());
   // Make sure query fields are present
   if (_queryDescriptor.getQueryFields().size()) {
                                 // Grid Field is present
      if (_repertoire->isGridRepertoire()) {
         if (queryFields[_gridIdx]->getField() == "Grid") {
            qr->_numFound++;
            std::auto_ptr<Queriable> aptrQueriable(new GridQueriable(_repertoire->getGrid()));
            qr->addQueriable(aptrQueriable);
         }
         else                    // query is for ConnectionSet
            setConnectionSetQueryResult(qr.get(), queryFields[_connectionSetPreIdx]->getField(), queryFields[_connectionSetPostIdx]->getField(), maxItem, minItem, searchSize);
      }
      else {
         if (queryFields[_subRepertoireIdx]->getField() != "") {
            std::list<Repertoire*>::const_iterator end = _subRepertoires.end();
            for (std::list<Repertoire*>::const_iterator iter = _subRepertoires.begin(); iter != end; iter++) {
               if (queryFields[_subRepertoireIdx]->getField() == (*iter)->getName()) {
                  qr->_numFound++;
                  if ((qr->_numFound>maxItem) || (qr->_numFound-minItem>searchSize)) {
                     qr->_searchCompleted=false;
                     iter=end;
                  }
                  else if (qr->_numFound>minItem) {
                     std::auto_ptr<Queriable> aptrQueriable(new RepertoireQueriable(*iter));
                     qr->addQueriable(aptrQueriable);
                  }
               }
            }
         }
         else                    // query is for ConnectionSet
            setConnectionSetQueryResult(qr.get(), queryFields[_connectionSetPreIdx]->getField(), queryFields[_connectionSetPostIdx]->getField(), maxItem, minItem, searchSize);
      }
   }
   else std::cerr<<"No query fields found in Repertoire!"<<std::endl;
   return qr;
}


Publisher* RepertoireQueriable::getQPublisher()
{
   return 0;
}


void RepertoireQueriable::setConnectionSetQueryResult (QueryResult* qr, std::string preName, std::string postName, int maxItem, int minItem, int searchSize)
{
   std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::const_iterator map_end = _connectionSetMap.end();
   for (std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::const_iterator iter = _connectionSetMap.begin(); iter != map_end; iter++) {
      if (preName == (*iter).first->getName()) {
         std::list<ConnectionSet*> cnxnSetList = (*iter).second;
         std::list<ConnectionSet*>::iterator list_end = cnxnSetList.end();
         for (std::list<ConnectionSet*>::iterator iter2 = cnxnSetList.begin(); iter2 != list_end; iter2++) {
                                 // no post name specified; just return the available posts in the QueryResult
            if (postName == "") {
               std::auto_ptr<Queriable> aptrQueriable(new ConnectionSetQueriable(*iter2));
               qr->addQueriable(aptrQueriable);
            }
            else {
               if (postName == (*iter2)->getPostPtr()->getName()) {
                  qr->_numFound++;
                  if ((qr->_numFound>maxItem) || (qr->_numFound-minItem>searchSize)) {
                     qr->_searchCompleted=false;
                     iter=map_end;
                  }
                  else if (qr->_numFound>minItem) {
                     std::auto_ptr<Queriable> aptrQueriable(new ConnectionSetQueriable(*iter2));
                     qr->addQueriable(aptrQueriable);
                  }
               }
            }
         }
         break;                  // no need to keep searching because pre's occur only once in the map
      }
   }
}


void RepertoireQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new RepertoireQueriable(*this));
}


RepertoireQueriable::~RepertoireQueriable()
{

}
