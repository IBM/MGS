#ifndef EpileptorNode_H
#define EpileptorNode_H

#include "Lens.h"
#include "CG_EpileptorNode.h"
#include "rndm.h"
#include<map>

#define Jirsa_et_al_2014
//#define Proix_et_al_2014
//#define Proix_et_al_2018

class EpileptorNode : public CG_EpileptorNode
{
   public:
      void initialize(RNG& rng);
      void updateDeltas(RNG& rng);
      void update(RNG& rng);
      virtual ~EpileptorNode();
      std::map<std::pair<int, int>, float> connectionMap;
};

#endif
