#include "Lens.h"
#include "ConnectNodeSetsByVolumeFunctor.h"
#include "CG_ConnectNodeSetsByVolumeFunctorBase.h"
#include "LensContext.h"
#include "Connector.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "ConnectionContext.h"
#include "SyntaxErrorException.h"
#include "ParameterSetDataItem.h"
#include "Simulation.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include "Coordinates.h"
#include "VecPrim.h"
#include "DimensionProducer.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"

#include <memory>

void ConnectNodeSetsByVolumeFunctor::userInitialize(LensContext* CG_c) 
{
}

void ConnectNodeSetsByVolumeFunctor::userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, String& center, float& radius, float& scale, Functor*& sourceOutAttr, Functor*& destinationInAttr) 
{
  /*
   * NOTE: The problem is one nodeset uses Tissue-Coordinate
   *       and the other nodeset uses Grid-Coordinate
   *       we need to convert a node in Grid-Coordinate to the location in TissueCoordinate
   *       (in Coordinates.h file
   *       the function assumes the center node of the grid has Tissue-Coordinate (0,0,0)
   *       )
   * center = "source" or "dest"  [which one should become the centered node]
   * radius = in micrometer, the radius of the sphere from the center node, and any nodes in the other nodeset
   *                     will be added to the connection
   * scale  = the length of each grid-element
   */
  CG_c->connectionContext->reset();
  ConnectionContext* cc = CG_c->connectionContext;
  Simulation* sim = CG_c->sim;
  cc->sourceSet = source;
  cc->destinationSet = destination;
  
  NodeSet* centering = source;
  NodeSet* sampling = destination;
  cc->current = ConnectionContext::_SOURCE;

  if (center=="source") {}
  else if (center=="dest") {
    centering = destination;
    sampling = source;
    cc->current = ConnectionContext::_DEST;
  }
  else {
     throw SyntaxErrorException(
	"ConnectNodeSetsByVolume : Center argument must be 'source' or 'dest'!");
  }
  
  Connector* lc;
  if (sim->isGranuleMapperPass()) {
    lc=_noConnector;
  } else if (sim->isCostAggregationPass()) {
    lc=_granuleConnector;
  } else if (sim->isSimulatePass()) {
    lc=_lensConnector;
  } else {
    throw SyntaxErrorException(
       "Error, ConnectNodeSetsByVolume : no connection context set!");
    exit(0);
  }

  std::vector<DataItem*> nullArgs;
  std::auto_ptr<DataItem> outAttrRVal;
  std::auto_ptr<DataItem> inAttrRVal;
  std::auto_ptr<DataItem> rval;
  ParameterSetDataItem *psdi;
  //cc->done = false;
  std::vector<NodeDescriptor*> _nodes;
  _nodes.clear();
  cc->destinationSet->getNodes(_nodes);
  cc->destinationNode = cc->destinationRefNode = *(_nodes.begin());
  std::vector<NodeDescriptor*> _nodesSource;
  _nodesSource.clear();
  cc->sourceSet->getNodes(_nodesSource);
  cc->sourceNode = cc->sourceRefNode = *(_nodesSource.begin());
  if (cc->sourceNode == 0 || not (sim->isSimulatePass()))
  {
    return;
  }
  sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
  psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
  if (psdi==0) {
    throw SyntaxErrorException(
       "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
  }
  cc->outAttrPSet = psdi->getParameterSet();
  
  destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
  psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
  if (psdi==0) {
    throw SyntaxErrorException(
       "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
  }
  cc->inAttrPSet = psdi->getParameterSet();

  std::vector<NodeDescriptor*> centerNodes;
  std::vector<NodeDescriptor*> sampleNodes;
  centering->getNodes(centerNodes);
  sampling->getNodes(sampleNodes);
  unsigned ctrs = centerNodes.size();
  unsigned smps = sampleNodes.size();

  std::vector<int> gridSize;
  gridSize = 
    centerNodes[0]->getGridLayerDescriptor()->getGrid()->getSize();
  
  std::vector<char> adjacency(ctrs*smps,0);  

  int myRank = sim->getRank();
  double criterion = radius*radius;
//#define DEBUG
#ifdef DEBUG
  int rank_2_use = 1;
  if (myRank == rank_2_use){
    std::cerr<< myRank<< " center-node size: " << ctrs << ", sample-node size: " << smps <<std::endl
      << std::flush;
    std::cerr<< "grid dim: " << gridSize[0]<<","<<gridSize[1]<<","<<gridSize[2]<<std::endl;
  }
#endif

  for (unsigned s=0; s<smps; ++s) {
    int smpRank = sim->getGranule(*sampleNodes[s])->getPartitionId();
    if (smpRank==myRank) {
      Node* n = sampleNodes[s]->getNode();
      DimensionProducer* dimProducer = dynamic_cast<DimensionProducer*>(n);
      if (dimProducer==0) {
	throw SyntaxErrorException(
	   "ConnectNodeSetsByVolume : Sampled node is not a DimensionProducer.");
	exit(0);
      }
      DimensionStruct* dims = dimProducer->CG_get_DimensionProducer_dimension();
      std::vector<double> smpCtr({dims->x, dims->y, dims->z});      
#ifdef DEBUG
      if (myRank == rank_2_use){
	std::cerr<< myRank<< " cpt-coord: " << dims->x << "," <<dims->y <<"," <<dims->z <<std::endl
	  << std::flush;
      }
#endif
      char* a = &adjacency[s*ctrs];
      for (unsigned c=0; c<ctrs; ++c, ++a) {
	std::vector<int> grdCtr;
	std::vector<double> volCtr;
	centerNodes[c]->getNodeCoords(grdCtr);
	if (grdCtr.size()!=gridSize.size() || gridSize.size()>3) {
	  throw SyntaxErrorException(
	     "ConnectNodeSetsByVolume: node coords have different dimensions than argument gridSize.");
	}
	//calculateRealCoordinatesCenter(grdCtr, scale, gridSize[0], gridSize[1], gridSize[2], volCtr);// grid's dimension is returned as (Z,Y,X)
	calculateRealCoordinatesCenterNTS(grdCtr, scale, gridSize[0], gridSize[1], gridSize[2], volCtr);// grid's dimension is returned as (X,Y,Z)
#ifdef DEBUG
      if (myRank == rank_2_use){
	std::cerr<< "-- nvu info: length=" << scale << ", index = "<< grdCtr[0] <<"," << grdCtr[1] << "," << grdCtr[2] <<std::endl;
	std::cerr<< myRank<< " NVU-center-coord: " << volCtr[0] << "," << volCtr[1] <<"," << volCtr[2] <<std::endl
	  << std::flush;
      }
#endif
	if (SqDist(&volCtr[0], &smpCtr[0])<criterion) *a=1;	
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &adjacency[0], adjacency.size(), MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

  for (unsigned s=0; s<smps; ++s) {
    char* a = &adjacency[s*ctrs];
    for (unsigned c=0; c<ctrs; ++c, ++a) { 
      NodeDescriptor* src=centerNodes[c];
      NodeDescriptor* dst=sampleNodes[s];
      if (cc->current==ConnectionContext::_DEST) {
	dst=centerNodes[c];
	src=sampleNodes[s];
      }
      if (*a) lc->nodeToNode(src, cc->outAttrPSet, dst, cc->inAttrPSet, sim);
    }
  }
#define DEBUG
#ifdef DEBUG
  if (myRank == 0){
    std::cerr << "Adjacency matrix of connection (row=source, col=dest)" << std::endl;
    for (unsigned s=0; s<smps; ++s) {
      for (unsigned c=0; c<ctrs; ++c) {
	std::cerr<< int(adjacency[s*ctrs+c]) ;
      }
      std::cerr << std::endl;
    }
  }
#endif
}

ConnectNodeSetsByVolumeFunctor::ConnectNodeSetsByVolumeFunctor() 
   : CG_ConnectNodeSetsByVolumeFunctorBase()
{
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

ConnectNodeSetsByVolumeFunctor::~ConnectNodeSetsByVolumeFunctor() 
{
}

void ConnectNodeSetsByVolumeFunctor::duplicate(std::auto_ptr<ConnectNodeSetsByVolumeFunctor>& dup) const
{
   dup.reset(new ConnectNodeSetsByVolumeFunctor(*this));
}

void ConnectNodeSetsByVolumeFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ConnectNodeSetsByVolumeFunctor(*this));
}

void ConnectNodeSetsByVolumeFunctor::duplicate(std::auto_ptr<CG_ConnectNodeSetsByVolumeFunctorBase>& dup) const
{
   dup.reset(new ConnectNodeSetsByVolumeFunctor(*this));
}

