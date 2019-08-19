#include "Lens.h"
#include "MahonUnitDataCollector.h"
#include "CG_MahonUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>


//#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
//#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()

void MahonUnitDataCollector::initialize(RNG& rng) 
{
  std::ostringstream sysCall;

  std::string output_dir(directory.c_str());
  if (output_dir.length() == 0)
    output_dir = "./";
  if (output_dir.back() != '/')
    output_dir.append("/");
  try {
    sysCall<<"mkdir -p "<<directory.c_str()<<" > /dev/null;";
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};

  {std::ostringstream os1;
    os1<< output_dir << "x1_"<<fileName;
    x1_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}
  

  {std::ostringstream os1;
    os1<< output_dir << "Spike_"<<fileName;
    spike_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}


  {std::ostringstream os1;
    os1<< output_dir << "x2_"<<fileName;
    x2_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}
  

  {std::ostringstream os1;
    os1<< output_dir << "x3_"<<fileName;
    x3_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}
  {std::ostringstream os1;
    os1<< output_dir << "timewindow_"<<fileName;
    x4_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}


  assert(rows.size()==cols.size());
  assert(cols.size()==spikes.size());
  assert(cols.size()==x1.size());
  assert(cols.size()==x2.size());
  assert(cols.size()==x3.size());


  /*

  std::map<unsigned, std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > > >  sorter;

  int sz=x1.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=std::pair<float*, std::pair<double*, bool*> >(V[j], std::pair<double*, bool*>(g[j],spikes[j]));
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  V.clear();
  spikes.clear();
  g.clear();
  std::map<unsigned, std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > > >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      V.push_back(miter2->second.first);
      spikes.push_back(miter2->second.second.second);
      g.push_back(miter2->second.second.first);
    }
  }


  */

  /*

  ShallowArray<double *> var1, var2, var3;
  ShallowArray<bool *> var4;
  //ShallowArray<int> index1;

  var1.increaseSizeTo(x1.size());
  var2.increaseSizeTo(x2.size());
  var3.increaseSizeTo(x3.size());
  var4.increaseSizeTo(spikes.size());

  
  var1 = x1;
  var2 = x2;
  var3 = x3;
  var4 = spikes;

  int sz=x1.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    //var1.push_back(x1[j]);
    //var2.push_back(x2[j]);
    //var3.push_back(x3[j]);
    //var4.push_back(spikes[j]);
    //sorter[rows[j]][cols[j]] = j;
    //sorter[rows[j]][cols[j]]=std::pair<float*, std::pair<double*, bool*> >(V[j], std::pair<double*, bool*>(g[j],spikes[j]));
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }


  unsigned sorter[mxrow+1][mxcol+1];
  for (int j=0; j<sz; ++j) 
    sorter[rows[j]][cols[j]] = j;

 
  
  assert(var1.size() == x1.size());

  
  x1.clear();
  x2.clear();
  x3.clear();
  spikes.clear();

  for (int j=0; j<=mxrow; j++)
    {
      for (int i=0; i<=mxcol; i++)
	{
	  const int k = sorter[j][i];
	  x1.push_back(var1[k]);
	  x2.push_back(var2[k]);
	  x3.push_back(var3[k]);
	  spikes.push_back(var4[k]);
	}
     }
  
  */

  /*

  std::map<unsigned, std::map<unsigned, unsigned> >:: iterator miter1, mend1 = sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, unsigned> :: iterator miter2, mend2 = miter1.end();
    for (miter2=miter1.begin(); miter2!=mend2; ++miter2) {
unsigned i = miter2
      x1.push_back(

    }
  }

  std::map<unsigned, std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > > >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      V.push_back(miter2->second.first);
      spikes.push_back(miter2->second.second.second);
      g.push_back(miter2->second.second.first);
    }
  }



  */


  

  //spike_output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;


  //file=new std::ofstream(fileName.c_str());
  //std::ofstream& output=*file;
  //output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;




}

void MahonUnitDataCollector::finalize(RNG& rng) 
{
  //dataCollection();
  *spike_file<<std::endl;
  *x1_file<<std::endl;
  *x2_file<<std::endl;
  *x3_file<<std::endl;
  *x4_file<< 0.0 << " " << ITER*deltaT <<std::endl;

  spike_file->close();
  x1_file->close();
  x2_file->close();
  x3_file->close();
  x4_file->close();

  {std::ofstream ofs;
  ofs.open ("weights.dat", std::ofstream::out | std::ofstream::app);
  ofs << -10000 << " " << -10000 << " " << -10000 << std::endl;
  ofs.close();}


}

void MahonUnitDataCollector::dataCollectionLFP(Trigger* trigger, NDPairList* ndPairList) 
{
  std::ofstream& output=*x3_file;
  ShallowArray<double*>::iterator iter=x3.begin(), end=x3.end();
  double tot = 0;
  for (;iter!=end; ++iter) tot+=(**iter); 
   
  output<<ITER*deltaT<<" "<<tot<<std::endl;
  

}

void MahonUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  {std::ofstream& output=*x1_file;
  output<<ITER*deltaT<<" ";//std::endl;
  ShallowArray<double*>::iterator iter=x1.begin(), end=x1.end();
  for (int col=0; iter!=end && col<maxoutnum; ++iter, ++col) {
    output<<**iter<<" ";

  }
  output<<std::endl;}


 
  {std::ofstream& output=*x2_file;
  output<<ITER*deltaT<<" ";//std::endl;
  ShallowArray<double*>::iterator iter=x2.begin(), end=x2.end();
  for (int col=0; iter!=end && col<maxoutnum; ++iter, ++col) {
    output<<**iter<<" ";
   
  }
  output<<std::endl;}

}




void MahonUnitDataCollector::dataCollectionSpike(Trigger* trigger, NDPairList* ndPairList) 
{
  std::ofstream& output=*spike_file;
  //output<<getSimulation().getIteration()<<" ";//std::endl;
  ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
  for (int col=0; iter!=end; ++iter, ++col) {
    if (**iter == true) 
      output<<getSimulation().getIteration()*deltaT<<" "<< col << std::endl;
    /*
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
    */
  }

  // output<<std::endl;

}




void MahonUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MahonUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

MahonUnitDataCollector::MahonUnitDataCollector() 
   : CG_MahonUnitDataCollector()
{
}

MahonUnitDataCollector::~MahonUnitDataCollector() 
{
  if (x1_file) delete x1_file; 
  if (spike_file) delete spike_file;
  if (x2_file) delete x2_file; 
  if (x3_file) delete x3_file; 
  if (x4_file) delete x4_file; 
}

void MahonUnitDataCollector::duplicate(std::unique_ptr<MahonUnitDataCollector>& dup) const
{
   dup.reset(new MahonUnitDataCollector(*this));
}

void MahonUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new MahonUnitDataCollector(*this));
}

void MahonUnitDataCollector::duplicate(std::unique_ptr<CG_MahonUnitDataCollector>& dup) const
{
   dup.reset(new MahonUnitDataCollector(*this));
}

