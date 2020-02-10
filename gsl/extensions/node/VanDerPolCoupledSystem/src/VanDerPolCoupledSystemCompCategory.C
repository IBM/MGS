#include "Lens.h"
#include "VanDerPolCoupledSystemCompCategory.h"
#include "NDPairList.h"
#include "CG_VanDerPolCoupledSystemCompCategory.h"

#include <fstream>
#include <unordered_set>
#include "StringUtils.h"
#include "FileUtils.h"

#define SHD getSharedMembers()

VanDerPolCoupledSystemCompCategory::VanDerPolCoupledSystemCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_VanDerPolCoupledSystemCompCategory(sim, modelName, ndpList),
    _checkfilename("VanDerPolCoupledSystemCheckopoint.bin")
{
}

void VanDerPolCoupledSystemCompCategory::outputWeightsShared(RNG& rng) 
{
}

void VanDerPolCoupledSystemCompCategory::initializeShared(RNG& rng) 
{
  /* this is important for get access to json-based data, and is loaded by the CompCat */
  auto it = _nodes.begin();
  auto end = _nodes.end();
  for (; it <= end; ++it) {
    it->setRealCompCategory(this);
  }
#if JSON_LIB == JSON_JSONXX
  assert(0);
  std::ifstream i_s(SHD.json_file.c_str());
  js.parse(StringUtils::to_string(i_s));
#elif JSON_LIB == JSON_NLOHMANN
  //json_stream.open(SHD.json_file.c_str(),  std::ifstream::in);
  if (! FileFolderUtils::isFileExist(std::string(SHD.json_file.c_str())))
  {
     std::cerr << "ERROR: File " << SHD.json_file << " not found\n";
     assert(0);
  }
  std::ifstream i_s(SHD.json_file.c_str());
  i_s >> js;
#endif
}

void VanDerPolCoupledSystemCompCategory::restore(RNG& rng) 
{
}

void VanDerPolCoupledSystemCompCategory::checkpoint(RNG& rng) 
{
}

