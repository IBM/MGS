#include "Lens.h"
#include "VanDerPolCoupledSystem.h"
#include "CG_VanDerPolCoupledSystem.h"
#include "rndm.h"
#ifdef HAVE_GPU
#include "CG_VanDerPolCoupledSystemCompCategory.h"
#endif
#include <cmath>

#include "Constants.h"
#include "NumberUtils.h"
#include "VanDerPolCoupledSystemCompCategory.h"
#define SHD getSharedMembers()

#if defined(USE_SIMULATION_INFO)
//use from shared time-step
#define dT (*(SHD.deltaT))
#define currentTime  (*(SHD.currentTime))
#else
//use from time-step passed explicitly for the node
#define dT ((SHD.dT))
#define currentTime (getSimulation().getIteration()*dT)
#endif


#ifdef HAVE_GPU

#define m  (_container->um_m[__index__])
#define x1  (_container->um_x1[__index__])
#define x2  (_container->um_x2[__index__])
#define alpha1  (_container->um_alpha1[__index__])
#define alpha2  (_container->um_alpha2[__index__])
#define W  (_container->um_W[__index__])
#define u  (_container->um_u[__index__])
#endif


void VanDerPolCoupledSystem::setRealCompCategory(VanDerPolCoupledSystemCompCategory* target) 
{_comCat=target;};
void VanDerPolCoupledSystem::initSolver() 
{
  /* define the size of the ODE system */
  /* 
   * x1 -> m of them
   * x2 -> m of them
   */
  const size_t N = 2*m; 

  double default_value = 1.0;
  // set initial values 
  x.resize(N, default_value);
  for (auto ii = 0; ii < m; ii++) {
    x[ii] = x1[ii];
    x[m+ii] = x2[ii];
  }
  adjust_size(N);
}

void VanDerPolCoupledSystem::initializeNode(RNG& rng) 
{
  _rng = &rng;
  // prepare I/O
  {
    ioStride = IO_STRIDE;
    prevIOIteration = 0;
    ioCounterFlush = HOW_OFTEN_TO_FLUSH; // how many I/O before flush to file

    Grid* g = this->getGridLayerDescriptor()->getGrid();
    GridLayerDescriptor* gl = this->getGridLayerDescriptor();
    std::string grid_name = g->getName(); // return the instance name of that grid
    grid_name = grid_name.substr(0, grid_name.length()-4);
    std::string gridlayer_name = gl->getName(); // return the instance name of the layer in that grid
    unsigned nodeIdx = getNode()->getNodeIndex();
    outFile = nullptr; 

    // create an object
    // find an entry
#if JSON_LIB == JSON_JSONXX
    {
      assert(0);
    //jsonxx::Object js = getRealCompCategory()->get<jsonxx::Object>("record");
    //if (js.has<jsonxx::Object>(grid_name)) {
    //  //found the grid instance
    //  auto v1 = js<jsonxx::Object>.get(grid_name);
    //  //auto v1 = js<jsonxx::Object>.get(grid_name).kv_map();
    //  //auto v1 = js[grid_name].get<std::unordered_map<std::string, json> >();
    //  // can be 
    //  //
    //  //"__real_area__": {"area_1": {"interneuron": [10, 2]}}, "pyramidal": [3]
    //  std::string real_area_key("__real_area__");
    //  if (v1.has(real_area_key))
    //  {
    //    //auto js = v1<jsonxx::Object>.get(real_area_key);
    //    auto js = js<jsonxx::Object>.get(real_area_key).kv_map();//map<string, Value*>
    //    for (auto it = js.begin(); it != js.end(); ++it) {
    //      std::string real_grid_name = it.key();
    //      //jsonxx::Object v1 = it.value().get<jsonxx::Object>();
    //      auto v1 = it.value().get<jsonxx::Object>().kv_map();
    //      //auto v1 = it.value().get<std::unordered_map<std::string, json>>();
    //      if (not v1.empty() and v1.count(gridlayer_name) > 0)
    //      {
    //        auto v2 = v1[gridlayer_name].get<jsonxx::Array>();
    //        //auto v2 = v1[gridlayer_name].get<std::vector<int>>();
    //        if(std::find(v2.begin(), v2.end(), nodeIdx) != v2.end()) {
    //          open_file(real_grid_name, gridlayer_name, nodeIdx);
    //        }
    //      }
    //    }
    //  }
    //  if (not v1.empty() and v1.count(gridlayer_name) > 0)
    //  {
    //    auto v2 = v1[gridlayer_name].get<std::vector<int> >();
    //    //auto v2 = v1[gridlayer_name].get<json>();
    //    if(std::find(v2.begin(), v2.end(), nodeIdx) != v2.end()) {
    //      open_file(grid_name, gridlayer_name, nodeIdx);
    //    }
    //  }
    //}
    }
#elif JSON_LIB == JSON_NLOHMANN
    {
      {
        auto js = getRealCompCategory()->getJSON();
        std::string name("RecordInterval");
        if (js.find(name) != js.end()) {
          ioStride = int(js["RecordInterval"].get<float>()/ dT);
        }
      }
    json js = getRealCompCategory()->getJSON()["record"];
    //json js;
    //getRealCompCategory()->getJSONStream() >> js;
    if (js.find(grid_name) != js.end()) {
      //found the grid instance
      auto v1 = js[grid_name].get<std::unordered_map<std::string, json> >();
      // can be 
      //
      //"__real_area__": {"area_1": {"interneuron": [10, 2]}}, "pyramidal": [3]
      if (v1.find("__real_area__") != v1.end())
      {
        json js = v1["__real_area__"].get<json>();
        for (json::iterator it = js.begin(); it != js.end(); ++it) {
          std::string real_grid_name = it.key();
          auto v1 = it.value().get<std::unordered_map<std::string, json>>();
          if (not v1.empty() and v1.count(gridlayer_name) > 0)
          {
            auto v2 = v1[gridlayer_name].get<std::vector<int>>();
            if(std::find(v2.begin(), v2.end(), nodeIdx) != v2.end()) {
              open_file(real_grid_name, gridlayer_name, nodeIdx);
            }
          }
        }
      }
      if (not v1.empty() and v1.count(gridlayer_name) > 0)
      {
        auto v2 = v1[gridlayer_name].get<std::vector<int> >();
        //auto v2 = v1[gridlayer_name].get<json>();
        if(std::find(v2.begin(), v2.end(), nodeIdx) != v2.end()) {
          open_file(grid_name, gridlayer_name, nodeIdx);
        }
      }
    }
    }
#endif
  }
  //resize data as array
#if defined(HAVE_GPU) 
  {
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      x1.resize_allocated_subarray(m, Array_Flat<int>::MemLocation::UNIFIED_MEM);
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      assert(0);
      _container->um_x1_num_elements[__index__] = m;
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      assert(0);
      _container->um_x1_num_elements[__index__] = 0;
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
      assert(0);
   #endif
  }
#else 
  if (x1.size() != m) x1.increaseSizeTo(m);
  if (x2.size() != m) x2.increaseSizeTo(m);
  if (alpha1.size() != m) alpha1.increaseSizeTo(m);
  if (alpha2.size() != m) alpha2.increaseSizeTo(m);
  if (W.size() != m*m) 
  {
    int prevSize = W.size();
    W.increaseSizeTo(m*m);
    for (unsigned ii = prevSize; ii < W.size(); ++ii)
      W[ii] = 0.0;
  }
  std::cout << "m = " << m << std::endl;
#endif
}

void VanDerPolCoupledSystem::initializeSolver(RNG& rng) 
{
  initSolver();
  write_file(true);
}

void VanDerPolCoupledSystem::operator() (const VanDerPolCoupledSystem_StateType &x, VanDerPolCoupledSystem_StateType &dxdt, double t)
{
  for (auto ii = 0; ii < m; ii++) {
    dxdt[ii] = alpha1[ii] * x[ii] * (1 - pow(x[ii], 2)) + alpha2[ii]*x[m+ii];
    for (auto jj = 0; jj < m; jj++)
    {
      dxdt[ii] += W[(m-1)*ii+jj] * x[jj];
    }
    dxdt[m+ii] = -1.0 * x[ii];
  }
}
void VanDerPolCoupledSystem::update(int ii)
{
#if defined(USE_SIMULATION_INFO)
  const double& t = currentTime;
#else
  double t = getSimulation().getIteration()*dT;
#endif
  do_step_small(this, x, t, dT, ii);

//#define VANDERPOL_SYNC_WITH_OTHERS
#if defined(VANDERPOL_SYNC_WITH_OTHERS)
  {
    //u = x[0];
  }
#endif
}
void VanDerPolCoupledSystem::update1(RNG& rng)
{
  write_file();
  update(1);
}

void VanDerPolCoupledSystem::update2(RNG& rng) 
{
  update(2);
}

void VanDerPolCoupledSystem::update3(RNG& rng) 
{
  update(3);
}

void VanDerPolCoupledSystem::update4(RNG& rng) 
{
  update(4);
}

VanDerPolCoupledSystem::~VanDerPolCoupledSystem() 
{
}

void VanDerPolCoupledSystem::open_file(std::string grid_name, std::string gridlayer_name, int nodeIdx)
{
  //std::string data_path(((String)getSharedMembers().data_path).c_str()); //#, getSharedMembers().data_path.size());
  //std::string data_path("./data");
  std::string data_path("./");
  std::string fileName;

  std::ostringstream os;
  std::ostringstream fieldName;
  {
    fileName = "rec";
    if (data_path.length() > 0)
      fileName = data_path + "/" + fileName;
    os << fileName 
      << "_" << nodeIdx 
      << "_" << grid_name
      << "_" << gridlayer_name;
      //<< "_rank"<< getSimulation().getRank() ;
    os << ".dat" << getSimulation().getRank();
    std::cout << "VDP output file: " << os.str() 
      << "   from grid " << grid_name 
      << "   of layer " << gridlayer_name 
      << std::endl;
    //outFile = new std::ofstream(os.str().c_str());
    outFile.reset(new std::ofstream(os.str().c_str()));
    outFile->precision(decimal_places);
    (*outFile) << "#Time" ;
    for (int ii = 0; ii < m; ii++)
    {
      fieldName.clear();
      fieldName.str("");
      fieldName << "x" << ii;
      (*outFile) << fieldDelimiter << fieldName.str(); 
    }
  }

}
/*
 * now (bool): force writing the data from the current timepoint to file immediately
 *            otherwise, it will write to its own pace, e.g. ioStride interval
 */
void VanDerPolCoupledSystem::write_file(bool now)
{
  int currentIter = getSimulation().getIteration();  // should also be (dt/2)
  bool write_file_now = false;
  if (currentIter >= prevIOIteration + ioStride or now)
  {
    write_file_now = true;
  }
  if (write_file_now)
  {
    if (outFile)
    {
      (*outFile) << std::endl;
      (*outFile) << std::fixed 
        << currentTime;
        for (int ii = 0; ii < m; ii++)
          (*outFile) << std::fixed 
            << fieldDelimiter << x[ii]; 
    }
    ioCounterFlush--;
    if (ioCounterFlush == 0)
    {
      if (outFile) (*outFile) << std::flush;
      ioCounterFlush = HOW_OFTEN_TO_FLUSH;
    }
  }

  if (currentIter >= prevIOIteration + ioStride)
  {
    prevIOIteration = currentIter;
  }
}
