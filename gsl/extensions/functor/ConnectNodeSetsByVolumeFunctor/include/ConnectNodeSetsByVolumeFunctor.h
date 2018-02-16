#ifndef ConnectNodeSetsByVolumeFunctor_H
#define ConnectNodeSetsByVolumeFunctor_H

#include "Lens.h"
#include "CG_ConnectNodeSetsByVolumeFunctorBase.h"
#include "LensContext.h"
#include <memory>

class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class ConnectNodeSetsByVolumeFunctor : public CG_ConnectNodeSetsByVolumeFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, String& center, float& radius, float& scale, ShallowArray< int >& gridSize, Functor*& sourceOutAttr, Functor*& destinationInAttr);
      ConnectNodeSetsByVolumeFunctor();
      virtual ~ConnectNodeSetsByVolumeFunctor();
      virtual void duplicate(std::auto_ptr<ConnectNodeSetsByVolumeFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ConnectNodeSetsByVolumeFunctorBase>& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};

#endif
