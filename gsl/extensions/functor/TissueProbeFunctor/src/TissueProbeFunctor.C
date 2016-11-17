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

#include "Lens.h"
#include "TissueProbeFunctor.h"
#include "CG_TissueProbeFunctorBase.h"
#include "LensContext.h"
#include "NodeSet.h"
#include "TissueFunctor.h"
#include "NodeDescriptor.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "Simulation.h"

#include <memory>
#include <algorithm>

void TissueProbeFunctor::userInitialize(LensContext* CG_c) 
{
}

std::auto_ptr<NodeSet> TissueProbeFunctor::userExecute(LensContext* CG_c) 
{
  std::auto_ptr<NodeSet> rval;
  NDPairList::iterator ndpiter = _tissueFunctor->_params->end(),
                       ndpend_reverse = _tissueFunctor->_params->begin();
  //NDPairList::iterator ndpiter = _tissueFunctor->getParams()->end(),
  //                     ndpend_reverse = _tissueParams->getParams()->begin();
  --ndpiter;
  --ndpend_reverse;

  if ((*ndpiter)->getName() == "PROBED")
  {//special-case
    if (CG_c->connectionContext->restart) {
      _grid = _tissueFunctor->doProbe(CG_c, _nodeDescriptors);
      assert(_grid);
      sort(_nodeDescriptors.begin(), _nodeDescriptors.end());
      CG_c->connectionContext->restart = false;
    }

    if (_nodeDescriptors.size()>0) { 
      std::vector<NodeDescriptor*> subtract, orig=_nodeDescriptors;
      NodeSet* ns = new NodeSet( _grid, _nodeDescriptors);
      ns->getNodes(subtract);
      std::vector<NodeDescriptor*>::iterator ndsit =
        std::set_difference(orig.begin(), orig.end(), 
            subtract.begin(), subtract.end(), 
            _nodeDescriptors.begin());
      _nodeDescriptors.resize(ndsit-_nodeDescriptors.begin());
      rval.reset(ns);
    }
    else if (_nodeDescriptors.size()==0) {
      NodeSet* ns=new NodeSet(_grid);
      ns->empty();
      rval.reset(ns);
      CG_c->connectionContext->done=true;
    }
  }
  else{
      _tissueFunctor->doProbe(CG_c, rval);
  }

  return rval;
}

TissueProbeFunctor::TissueProbeFunctor() 
   : CG_TissueProbeFunctorBase(), _tissueFunctor(0), _grid(0)
{
}

TissueProbeFunctor::~TissueProbeFunctor() 
{
}

TissueProbeFunctor::TissueProbeFunctor(TissueProbeFunctor* tpf)
  : CG_TissueProbeFunctorBase(), _tissueFunctor(tpf->_tissueFunctor), 
    _grid(tpf->_grid), _nodeDescriptors(tpf->_nodeDescriptors)
{
}


void TissueProbeFunctor::duplicate(std::auto_ptr<TissueProbeFunctor>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

void TissueProbeFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

void TissueProbeFunctor::duplicate(std::auto_ptr<CG_TissueProbeFunctorBase>& dup) const
{
   dup.reset(new TissueProbeFunctor(*this));
}

