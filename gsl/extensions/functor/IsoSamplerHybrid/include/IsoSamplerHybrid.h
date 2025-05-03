// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef IsoSamplerHybrid_H
#define IsoSamplerHybrid_H

#include "Mgs.h"
#include "CG_IsoSamplerHybridBase.h"
#include "GslContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class IsoSamplerHybrid : public CG_IsoSamplerHybridBase
{
   public:
      void userInitialize(GslContext* CG_c);
      void userExecute(GslContext* CG_c);
      IsoSamplerHybrid();
      virtual ~IsoSamplerHybrid();
      virtual void duplicate(std::unique_ptr<IsoSamplerHybrid>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_IsoSamplerHybridBase>&& dup) const;

   private:
      bool _done;
      int _nbrNodes;
      std::vector<NodeDescriptor*> _srcNodes;
      std::vector<NodeDescriptor*> _dstNodes;
      int _nodeIndex;
};

#endif
