#ifndef IsoSamplerHybrid_H
#define IsoSamplerHybrid_H

#include "Lens.h"
#include "CG_IsoSamplerHybridBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class IsoSamplerHybrid : public CG_IsoSamplerHybridBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c);
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
