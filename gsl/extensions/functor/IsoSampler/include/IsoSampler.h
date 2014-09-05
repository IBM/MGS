#ifndef IsoSampler_H
#define IsoSampler_H

#include "Lens.h"
#include "CG_IsoSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class IsoSampler : public CG_IsoSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c);
      IsoSampler();
      virtual ~IsoSampler();
      virtual void duplicate(std::auto_ptr<IsoSampler>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_IsoSamplerBase>& dup) const;

   private:
      bool _done;
      int _nbrNodes;
      std::vector<NodeDescriptor*> _srcNodes;
      std::vector<NodeDescriptor*> _dstNodes;
      int _nodeIndex;
};

#endif
