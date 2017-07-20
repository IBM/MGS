#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <sstream>
#include <istream>
#include <algorithm>
#include "../nti/include/Params.h"

void genMatrix()
{
  // make sure given 2 vectors
  // generate an matrix whose element is a vector with elements from from
  // first,and then second vector,
  //  in that order
  // initialize test data
  std::vector<std::vector<int> > vect_values;
  unsigned int sz = 3;
  int total_vect = 1;
  for (unsigned int j = 0; j < sz; ++j)
  {
    if (j == 0)
    {
      std::vector<int> values;
      values.push_back(2);
      values.push_back(3);
      vect_values.push_back(values);
    }
    else if (j == 1)
    {
      std::vector<int> values;
      values.push_back(4);
      values.push_back(5);
      values.push_back(6);
      vect_values.push_back(values);
    }
    else if (j == 2)
    {
      std::vector<int> values;
      values.push_back(1);
      values.push_back(6);
      values.push_back(9);
      vect_values.push_back(values);
    }
    total_vect *= vect_values.back().size();
  }

  // body code
  std::vector<unsigned int*> v_ids;
  unsigned int** pids = new unsigned int* [total_vect];
  for (int jj = 0; jj < total_vect; ++jj)
  {
    pids[jj] = new unsigned int[sz]();
    v_ids.push_back(pids[jj]);
  }
  for (unsigned int jj = 0; jj < sz; jj++)
  {
    int num2clone = 1;
    for (unsigned int xx = jj + 1; xx < sz; xx++)
      num2clone *= vect_values[xx].size();
    int gap = num2clone * vect_values[jj].size();

    std::cout << "gap=" << gap << ";num2clone= " << num2clone << std::endl;

    for (unsigned int kk = 0; kk < vect_values[jj].size(); kk++)
    {
      for (int xx = (num2clone) * (kk); xx < total_vect; xx += gap)
      {
        std::vector<unsigned int*>::iterator iter,
            iterstart = v_ids.begin() + xx,
            iterend = v_ids.begin() + xx + num2clone - 1;
        for (iter = iterstart; iter <= iterend; iter++)
          (*iter)[jj] = vect_values[jj][kk];
      }
    }
  }

  std::vector<unsigned int*>::iterator iter, iterstart = v_ids.begin(),
                                             iterend = v_ids.end();
  for (iter = iterstart; iter < iterend; iter++)
  {
    for (unsigned int i = 0; i < sz; i++) std::cout << (*iter)[i];
    std::cout << std::endl;
  }

  /*std::vector<std::vector<int> >::iterator viter = vect_values.begin(),
          viter_end = vect_values.end();
  for (; viter < viter_end; viter++)
  {
  }
  */
}

void genVector()
{
  // check the code to read in and generate the vector
  // NOTE: make sure all the following tests are correct
  // [1,3:5] --> generate 1,3,4,5
  //std::string mystring = "[1,3:5]"; //expect 1,3,4,5
  //std::string mystring = "[1]";     //expect 1
  //std::string mystring = "[1:2,4:6]";//expect 1,2,4,5,6
  //std::string mystring = "[1,2]";  //expect 1,2
  //std::string mystring = "[1,2,]";   //expect 1,2
  std::string mystring = "[3:6] 0";   //expect 3,4,5,6
  //std::string mystring = "[3:5]";

  std::cout <<  "content: " << mystring << std::endl;

  std::string filename = "outname.txt";
  {
    FILE* fPF = fopen(filename.c_str(), "w");
    fprintf(fPF, mystring.c_str(), "%s");
    fclose(fPF);
  }

  {
    FILE* fPF = fopen(filename.c_str(), "r");
    char input[2000];
    /*
     fgets(input, 2000, fPF);
     std::cout << input << std::endl;
    */

    Params params;
    std::vector<int> values;
    params.getListofValues(fPF, values);

    fclose(fPF);
    std::vector<int>::iterator iter=values.begin(), iend = values.end();
    for (; iter< iend;iter++)
      std::cout << *iter << ",";
    std::cout << std::endl;
  }
}

// NOTE:
//
// mpic++ -std=c++11 main_genvect.cpp ../nti/obj/Params.o  ../nti/obj/SegmentDescriptor.o ../common/obj/StringUtils.o ../common/obj/NumberUtils.o -lgmp
int main()
{  // check the code to generate an array of vectors from the range
  // [1:5] 5 --> would give 5 vectors (1,5)...,(5,5)
  // [1:3,5] 5 --> would give 4 vectors (1,5),(2,5),(3,5),(5,5)
  genVector();
}
// cd ../nti/;make clean; make debug=yes; cd -; mpic++ -g main_genVect.cpp ../nti/obj/Params.o ../nti/obj/SegmentDescriptor.o ../common/obj/StringUtils.o
