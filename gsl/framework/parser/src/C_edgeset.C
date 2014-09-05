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

#include "C_edgeset.h"
#include "C_edgeset_extension.h"
#include "C_nodeset.h"
#include "C_declarator.h"
#include "LensContext.h"
#include "Node.h"
#include "NodeSet.h"
#include "NodeSetDataItem.h"
#include "Grid.h"
#include "Repertoire.h"
#include "Edge.h"
#include "ConnectionSet.h"
#include "EdgeSet.h"
#include "EdgeSetDataItem.h"
#include "Simulation.h"
#include "LensConnector.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

#include <vector>

void C_edgeset::internalExecute(LensContext *c)
{
   if (!c->sim->isEdgeRelationalDataEnabled()) {
     std::cerr << "Edge Relational Data is disabled on simulation! Enable for use of EdgeSet in specification file." << std::endl;
     exit(-1);
   }
   if (_edgesetExtension) _edgesetExtension->execute(c);
   if (_pre) _pre->execute(c);
   if (_post) _post->execute(c);
   if (_edgesetDeclarator)_edgesetDeclarator->execute(c);
   if (_dec1) _dec1->execute(c);
   if (_dec2) _dec2->execute(c);
   if (_edgeset1) _edgeset1->execute(c);
   if (_edgeset2) _edgeset2->execute(c);

   const NodeSetDataItem* nsdi1 = 0;
   const NodeSetDataItem* nsdi2 = 0;
   const EdgeSetDataItem* esdi1 = 0;
   const EdgeSetDataItem* esdi2 = 0;
   NodeSet* preNS = 0;
   NodeSet* postNS = 0;

   if (_dec1 && _dec2) {
      DataItem* di = c->symTable.getEntry(_dec1->getName());
      nsdi1 = dynamic_cast<const NodeSetDataItem*>(di);
      if (nsdi1 == 0) {
         esdi1 = dynamic_cast<const EdgeSetDataItem*>(di);
         if (esdi1 == 0) {
	    std::string mes = "dynamic cast of DataItem to NodeSetDataItem and to EdgeSetDataItem failed";
	    throwError(mes);
         }
      }
      di = c->symTable.getEntry(_dec2->getName());
      nsdi2 = dynamic_cast<const NodeSetDataItem*>(di);
      if (nsdi2 == 0) {
         esdi2 = dynamic_cast<const EdgeSetDataItem*>(di);
         if (esdi2 == 0) {
	    std::string mes = "dynamic cast of DataItem to EdgeSetDataItem failed";
	    throwError(mes);
         }
      }
      if (nsdi1 && nsdi2) {
	 preNS = nsdi1->getNodeSet();
	 postNS = nsdi2->getNodeSet();
      }
   } else if (_pre && _post) {
      preNS = _pre->getNodeSet();
      postNS = _post->getNodeSet();
   }

   if (_edgesetDeclarator) {
      const EdgeSetDataItem* esdi = 
	 dynamic_cast<const EdgeSetDataItem*>(
	    c->symTable.getEntry(_edgesetDeclarator->getName()) );
      if (esdi == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to EdgeSetDataItem failed";
	 throwError(mes);
      }
      _edgeset = new EdgeSet(esdi->getEdgeSet());
   }
   else if ((nsdi1 && nsdi2) || (_pre && _post)) {  
      _edgeset = new EdgeSet();
      Grid* preGrid = preNS->getGrid();
      Grid* postGrid = postNS->getGrid();
      LensConnector lc;
      Repertoire* parent = lc.findLeastCommonRepertoire(preGrid, postGrid);
      const std::vector<GridLayerDescriptor*>& preLayers = preNS->getLayers(),
	 postLayers = postNS->getLayers();
      int preSize = preLayers.size();
      int postSize = postLayers.size();
      // Careful this copying is needed!!!
      std::vector<Edge*> edges;
      // These are for the state machine down below
      bool noErase; 
      bool eraseState;
      std::vector<Edge*>::iterator edgeIter, edgeEnd, eraseBegin, eraseEnd;
      for (int i=0; i<preSize; ++i) {
         for (int j=0; j<postSize; ++j) {
	    edges = *(parent->getConnectionSet(preLayers[i], postLayers[j]));
	    // Mark the point where the first one to be deleted is found
	    // Iterate until smthng that doesn't have to be deleted.
	    // Delete region.
            do {
	       noErase = true;
	       eraseState = false;
	       edgeIter = edges.begin();
	       edgeEnd = edges.end();
	       for (; edgeIter != edgeEnd; ++edgeIter) {
		  Edge* e = (*edgeIter);
		  if (eraseState) {
		     if (preNS->contains(e->getPreNode()) 
			 && postNS->contains(e->getPostNode())) {
			eraseEnd = edgeIter;
			edges.erase(eraseBegin, eraseEnd);
 			noErase = false;
			break;
		     } 
		  } else {
		     if (!preNS->contains(e->getPreNode())
			 || !postNS->contains(e->getPostNode())) {
			eraseBegin = edgeIter;
			eraseState = true;
		     } 
		  }
	       }
	       if (eraseState && noErase) {
		  edges.erase(eraseBegin, edgeEnd);
		  noErase = false;
	       } 
	    } while (!noErase);
            _edgeset->addEdges(&edges);
         }
      }
   }
   else if (_edgeset1 && _edgeset2) {
      _edgeset = new EdgeSet(_edgeset1->getEdgeSet());
      _edgeset->addEdges(_edgeset2->getEdgeSet());
   }
   else if (esdi1 && esdi2) {
      _edgeset = new EdgeSet(esdi1->getEdgeSet());
      _edgeset->addEdges(esdi2->getEdgeSet());
   }

   // now prune the edgeset if edgeSetExtension is specified
   if (_edgesetExtension) {
      std::vector<int> const & indices = _edgesetExtension->getIndices();
      std::string const & type = _edgesetExtension->getEdgeType();
      if (indices.size() > 0) {
         std::vector<Edge*> edges = _edgeset->getEdges();
         std::vector<int>::const_iterator indexIter = indices.begin();
         std::vector<int>::const_iterator indexEnd = indices.end();
         int idx = (*indexIter);
         std::vector<Edge*>::iterator edgeIter, edgeBegin = edges.begin(),
	    edgeEnd = edges.end();
         for (edgeIter = edgeBegin; edgeIter != edgeEnd; ++edgeIter) {
            if (edgeIter-edgeBegin != idx) edges.erase(edgeIter);
            else idx = *(++indexIter);
            if (indexIter == indexEnd) break;
         }
         _edgeset->reset();
         _edgeset->addEdges(&edges);
      }
      if (type.size() > 0) {
         EdgeSet* es = new EdgeSet();
         std::vector<Edge*> edges = _edgeset->getEdgeTypeSet(type);
         es->addEdges(&edges);
         delete _edgeset;
         _edgeset = es;
      }
   }
}

C_edgeset::C_edgeset(const C_edgeset& rv)
   : C_production(rv), _edgesetExtension(0), _pre(0), _post(0), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(0), 
     _edgeset2(0)
{
   if (rv._edgesetExtension) {
      _edgesetExtension = rv._edgesetExtension->duplicate();
   }
   if (rv._pre) {
      _pre = rv._pre->duplicate();
   }
   if (rv._post) {
      _post = rv._post->duplicate();
   }
   if (rv._edgesetDeclarator) {
      _edgesetDeclarator = rv._edgesetDeclarator->duplicate();
   }
   if (rv._edgeset) {
      _edgeset = new EdgeSet(rv._edgeset);
   }
   if (rv._dec1) {
      _dec1 = rv._dec1->duplicate();
   }
   if (rv._dec2) {
      _dec2 = rv._dec2->duplicate();
   }
   if (rv._edgeset1) {
      _edgeset1 = rv._edgeset1->duplicate();
   }
   if (rv._edgeset2) {
      _edgeset2 = rv._edgeset2->duplicate();
   }
}


C_edgeset::C_edgeset(C_declarator* esd, C_edgeset_extension* ese, 
		     SyntaxError * error)
   : C_production(error), _edgesetExtension(ese), _pre(0), _post(0), 
     _edgesetDeclarator(esd), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(0), 
     _edgeset2(0)
{
}


C_edgeset::C_edgeset(C_declarator* dec1, C_declarator* dec2, 
		     SyntaxError * error)
   : C_production(error), _edgesetExtension(0), _pre(0), _post(0), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(dec1), _dec2(dec2), 
     _edgeset1(0), _edgeset2(0)
{
}


C_edgeset::C_edgeset(C_declarator* dec1, C_declarator* dec2, 
		     C_edgeset_extension* ese, SyntaxError * error)
   : C_production(error), _edgesetExtension(ese), _pre(0), _post(0), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(dec1), _dec2(dec2), 
     _edgeset1(0), _edgeset2(0)
{
}


C_edgeset::C_edgeset(C_nodeset* pre, C_nodeset* post, SyntaxError * error)
   : C_production(error), _edgesetExtension(0), _pre(pre), _post(post), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(0), 
     _edgeset2(0)
{
}


C_edgeset::C_edgeset(C_nodeset* pre, C_nodeset* post, C_edgeset_extension* ese,
		     SyntaxError * error)
   : C_production(error), _edgesetExtension(ese), _pre(pre), _post(post), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(0), 
     _edgeset2(0)
{
}


C_edgeset::C_edgeset(C_edgeset* es1, C_edgeset* es2, SyntaxError * error)
   : C_production(error), _edgesetExtension(0), _pre(0), _post(0), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(es1), 
     _edgeset2(es2)
{
}


C_edgeset::C_edgeset(C_edgeset* es1, C_edgeset* es2, C_edgeset_extension* ese, 
		     SyntaxError * error)
   : C_production(error), _edgesetExtension(ese), _pre(0), _post(0), 
     _edgesetDeclarator(0), _edgeset(0), _dec1(0), _dec2(0), _edgeset1(es1), 
     _edgeset2(es2)
{
}


C_edgeset* C_edgeset::duplicate() const
{
   return new C_edgeset(*this);
}


C_edgeset::~C_edgeset()
{
   delete _edgesetExtension;
   delete _pre;
   delete _post;
   delete _edgesetDeclarator;
   delete _edgeset;
   delete _dec1;
   delete _dec2;
   delete _edgeset1;
   delete _edgeset2;
}

void C_edgeset::checkChildren() 
{
   if (_edgesetExtension) {
      _edgesetExtension->checkChildren();
      if (_edgesetExtension->isError()) {
         setError();
      }
   }
   if (_pre) {
      _pre->checkChildren();
      if (_pre->isError()) {
         setError();
      }
   }
   if (_post) {
      _post->checkChildren();
      if (_post->isError()) {
         setError();
      }
   }
   if (_edgesetDeclarator) {
      _edgesetDeclarator->checkChildren();
      if (_edgesetDeclarator->isError()) {
         setError();
      }
   }
   if (_dec1) {
      _dec1->checkChildren();
      if (_dec1->isError()) {
         setError();
      }
   }
   if (_dec2) {
      _dec2->checkChildren();
      if (_dec2->isError()) {
         setError();
      }
   }
   if (_edgeset1) {
      _edgeset1->checkChildren();
      if (_edgeset1->isError()) {
         setError();
      }
   }
   if (_edgeset2) {
      _edgeset2->checkChildren();
      if (_edgeset2->isError()) {
         setError();
      }
   }
} 

void C_edgeset::recursivePrint() 
{
   if (_edgesetExtension) {
      _edgesetExtension->recursivePrint();
   }
   if (_pre) {
      _pre->recursivePrint();
   }
   if (_post) {
      _post->recursivePrint();
   }
   if (_edgesetDeclarator) {
      _edgesetDeclarator->recursivePrint();
   }
   if (_dec1) {
      _dec1->recursivePrint();
   }
   if (_dec2) {
      _dec2->recursivePrint();
   }
   if (_edgeset1) {
      _edgeset1->recursivePrint();
   }
   if (_edgeset2) {
      _edgeset2->recursivePrint();
   }
   printErrorMessage();
} 
