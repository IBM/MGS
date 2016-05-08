#ifndef __FILEFOLDERUTILS_H__
#define __FILEFOLDERUTILS_H__

#include <string>
#include <vector>

class FileFolderUtils
{
  public:
  static void isFolderExist(const std::string& str,
                            const std::string& fileExtension = ".dat,.txt");
};

#endif
