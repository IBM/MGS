#include "../nti/include/Params.h"
#include <iostream>
#include <string>
#include <iomanip>

//x = row, y=col
//WIDTH=#col, HEIGHT=#row
#define Map1Dindex(x,y, WIDTH) ((y)+(x)*(WIDTH))

#define Find2Dindex(x,y, i, WIDTH) \
		do{\
			(y) = (i) % (WIDTH);\
			(x) = (i) / (WIDTH);\
		}while (0)

// cd ../nti/;make clean; make debug=yes; cd -; mpic++ -g main_parseMarkov.cpp ../nti/obj/Params.o ../nti/obj/SegmentDescriptor.o ../common/obj/StringUtils.o
// cd ../nti/;make clean; make debug=yes; cd -; mpic++ -g main_parseMarkov.cpp ../nti/obj/Params.o ../nti/obj/SegmentDescriptor.o ../common/obj/StringUtils.o -lgmp -lgmpxx -std=c++11
	std::string fname="RYR_Markov_Williams2012.conf";
	float* matChannelRateConstant;
	int numChanStates = 0;
	int*vOpenStates;
	int initialstate;
void parseMarkov(Params param)
{
	param.readMarkovModel(fname, matChannelRateConstant, numChanStates, vOpenStates, initialstate);
	std::cout<< "numChanStates = " << numChanStates << std::endl;
	assert(numChanStates == 2);

	std::cout << "Vector of conduct=1,non-conduct=0 states: " << std::endl;
	for (int ii=0; ii < numChanStates; ii++) 
 	  std::cout << vOpenStates[ii] << " ";
  std::cout	<< std::endl;
	assert(vOpenStates[0]==0);
	assert(vOpenStates[1]==1);

	std::cout<< "initial state= " << initialstate<< std::endl;
	assert(initialstate==1);
	for (int ii=0; ii < numChanStates; ii++)
	{
		for (int jj=0; jj < numChanStates; jj++)
			//std::cout << std::setw(5) << matChannelRateConstant[ii][jj] <<",";
			std::cout << std::setw(5) << matChannelRateConstant[Map1Dindex(ii,jj, numChanStates)] <<",";
		std::cout << std::endl;
	}
}

void genCluster(Params param)
{
	int numChan = 50;
	int* vClusterNumOpenChan;
	int numClusterStates;
	int* StateVector;
	int maxNumNeighbors;
	long * matK_channelstate_fromto;
	int* matK_indx;
	param.setupCluster(
			matChannelRateConstant,
			numChan, numChanStates, vOpenStates, 
			numClusterStates, 
			StateVector, 
			vClusterNumOpenChan,  
			maxNumNeighbors,
			matK_channelstate_fromto, matK_indx
			);
	std::cin.get();
	std::cout << "Cluster #states = " << numClusterStates << std::endl;
	for (int ii =0; ii < numClusterStates; ii++)
	{
		for (int jj = 0; jj < numChanStates; jj++)
			std::cout << StateVector[Map1Dindex(ii,jj,numChanStates)] << ",";
		std::cout << std::endl;

	}
	std::cout << "Max #neighbors = " << maxNumNeighbors << std::endl;
	std::cout << "Info: #opening channels at each cluster-state\n";
	for (int ii =0; ii < numClusterStates; ii++)
	{
		std::cout << vClusterNumOpenChan[ii] << ", ";
	}
	std::cout << 	std::endl;
	std::cout << "matK_channel_fromto:\n";
	for (int ii =0; ii < numClusterStates; ii++)
	{
		for (int jj = 0; jj < numChanStates; jj++)
			std::cout << matK_channelstate_fromto[Map1Dindex(ii,jj,numChanStates)] << ",";
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "matK_indx:\n";
	for (int ii =0; ii < numClusterStates; ii++)
	{
		for (int jj = 0; jj < numChanStates; jj++)
			std::cout << matK_indx[Map1Dindex(ii,jj,numChanStates)] << ",";
		std::cout << std::endl;
	}
}

int main(){
	// parse a Markov file for single channel
	
	Params param;
	parseMarkov(param);
	

	genCluster(param);

	//free memory
	/*for (int ii=0; ii < numChanStates; ii++)
	{
		delete matChannelRateConstant[ii];
	}
	delete []matChannelRateConstant;
	*/
	delete matChannelRateConstant;
	delete vOpenStates;
	return 1;
}
