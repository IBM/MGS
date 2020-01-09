#ifndef VanDerPolCoupledSystemCompCategory_H
#define VanDerPolCoupledSystemCompCategory_H

#include "Lens.h"
#include "CG_VanDerPolCoupledSystemCompCategory.h"

#include <list>
#include <string>
#include <typeinfo>
#include "Constants.h"

#if JSON_LIB == JSON_JSONXX
#include "jsonxx.h"
//using namespace jsonxx;
#elif JSON_LIB == JSON_NLOHMANN
#include "../nlohmann/json.hpp"
// for convenience
using json = nlohmann::json;
#endif

class NDPairList;

class VanDerPolCoupledSystemCompCategory : public CG_VanDerPolCoupledSystemCompCategory
{
   public:
      VanDerPolCoupledSystemCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void outputWeightsShared(RNG& rng);
      void initializeShared(RNG& rng);
      void restore(RNG& rng);
      void checkpoint(RNG& rng);
#if JSON_LIB == JSON_JSONXX
      jsonxx::Object& getJSON() { return js; };

#elif JSON_LIB == JSON_NLOHMANN
      json& getJSON() { return js; };
#endif
   private:
      std::string _checkfilename;
#if JSON_LIB == JSON_JSONXX
      jsonxx::Object js;

#elif JSON_LIB == JSON_NLOHMANN
      json js;
#endif
};

#endif
