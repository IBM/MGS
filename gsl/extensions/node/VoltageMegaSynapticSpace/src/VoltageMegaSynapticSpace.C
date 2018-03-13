#include "Lens.h"
#include "VoltageMegaSynapticSpace.h"
#include "CG_VoltageMegaSynapticSpace.h"
#include "rndm.h"
#include "CG_VoltageAdapter.h"
#include "Coordinates.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"

void VoltageMegaSynapticSpace::produceInitialVoltage(RNG& rng) 
{
   _numInputs = Vm.size();
}

void VoltageMegaSynapticSpace::produceVoltage(RNG& rng) 
{
}

void VoltageMegaSynapticSpace::computeState(RNG& rng) 
{
   //aggregate all ShallowArray<dyn_var_t*> Vm
   //to produce LFP
   LFP = 0.0; 
   //std::for_each(Vm.begin(), Vm.end(), [&])
   for (auto& n : Vm)
      LFP +=  *n;
   LFP /= _numInputs;
}

bool VoltageMegaSynapticSpace::isInRange(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageMegaSynapticSpaceInAttrPSet* CG_inAttrPset, CG_VoltageMegaSynapticSpaceOutAttrPSet* CG_outAttrPset) 
{
   CG_VoltageAdapter* cg_node = dynamic_cast<CG_VoltageAdapter*>(CG_node->getNode());
   if (not cg_node)
   {
      std::cerr << "ERROR: You need to connect VoltageAdapter to VoltageMegaSynapticSpace"<< std::endl;
      assert(0);
   }
   DimensionStruct* dimension = cg_node->CG_get_DimensionProducer_dimension();
   std::vector<double> coords {dimension->x, dimension->y, dimension->z};

   std::vector<int> indexCoordinateFrom;
   //Connexon
   std::vector<int> sizes = getGridLayerDescriptor()->getGrid()->getSize();
   calculateIndexCoordinates(coords, getSharedMembers().L0, 
         getSharedMembers().gridSize[0],
         getSharedMembers().gridSize[1],
         getSharedMembers().gridSize[2],
         indexCoordinateFrom);
   //MegaSC
   NodeBase* nodebase = dynamic_cast<NodeBase*>(CG_node->getNode());
   std::vector<int> indexCoordinateTo;
   nodebase->getNodeCoords(indexCoordinateTo);
   return (indexCoordinateFrom == indexCoordinateTo);
}
VoltageMegaSynapticSpace::~VoltageMegaSynapticSpace() 
{
}

