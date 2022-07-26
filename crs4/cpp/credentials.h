// Copyright 2021-2 CRS4
// 
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CREDENTIALS_H
#define CREDENTIALS_H

#include<string>

class Credentials{
public:
  std::string password;
  std::string username;
  Credentials(std::string p, std::string u) : password(p), username(u) {}
};

#endif
