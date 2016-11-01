
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <sstream>
#include <algorithm>
#include "../nti/include/Params.h"

int main()
{  // check the default value
  Params param;
  {
    std::string fname = "params/DetParams.par";
    param.readDevParams(fname);
  }

  {
    std::string fname = "params/ChanParams.par";
    param.readChanParams(fname);
  }
  {
    std::string fname = "params/ChanParams-Hay2011.par";
    param.readChanParams(fname);
  }
}
