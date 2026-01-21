// bin/prem -i data/prem.200.noiso -o data/prem -order 2 -alg 1 -s 150e3-600e3
#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<double> extractLayerBoundaries(const std::string &fileName,
                                           double &R) {
  std::vector<double> radii;
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << fileName << std::endl;
    return radii;
  }

  std::string line;
  double previousRadius = -1.0;

  int lineCount = 0;
  while (std::getline(file, line)) {
    if (lineCount < 3) {
      lineCount++;
      continue;
    }

    std::istringstream iss(line);
    double radius, density, pWave, sWave, bulkM, shearM;
    if (iss >> radius >> density >> pWave >> sWave >> bulkM >> shearM) {
      if (std::abs(radius - previousRadius) < 1e-3) {
        radii.push_back(radius);
      }
      previousRadius = radius;
    }
  }
  radii.push_back(previousRadius);
  if (R < 0) R = radii.back();
  double fac = 1.2;
  double radius_max = fac * R;
  radii.push_back(radius_max);
  for (double &r : radii) {
    r /= R;
  }

  file.close();
  return radii;
}

void createConcentricSphericalLayers(
    const std::vector<double> &radii,
    double meshSizeMin, double meshSizeMax, int elementOrder, int algorithm,
    const std::string &outputFileName) {
  int numLayers = radii.size();
  if (numLayers < 1) {
    std::cerr << "Error: There should be at least one layer." << std::endl;
    return;
  }
  // Initialize Gmsh
  gmsh::initialize();
  gmsh::model::add("ConcentricSphericalLayers");

  // Set mesh size options
  gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
  gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);

  int layerTag = 1;
  int surfaceTag = 1;
  for (int i = 0; i < numLayers; ++i) {
    gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
  }
  gmsh::model::occ::synchronize();
  gmsh::model::addPhysicalGroup(3, {1}, layerTag);
  gmsh::model::setPhysicalName(3, layerTag, "layer_1");
  std::vector<std::pair<int, int>> surfaceEntities;
  gmsh::model::getBoundary({{3, 1}}, surfaceEntities, false, false, false);
  std::pair<int, int> surface = surfaceEntities[0];
  if (surface.first == 2) {
    gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
    gmsh::model::setPhysicalName(2, surfaceTag, "surface_1");
  }
  for (int i = 1; i < numLayers; ++i) {
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int>>> ovv;

    gmsh::model::occ::cut({{3, i + 1}}, {{3, i}}, ov, ovv, -1, false, false);
    gmsh::model::occ::synchronize();

    std::vector<int> volumeTags;
    for (const auto &entity : ov) {
      volumeTags.push_back(entity.second);  
    }
    ++layerTag;
    gmsh::model::addPhysicalGroup(3, volumeTags, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "layer_" + std::to_string(i + 1));
    for (const auto &volumeTag : volumeTags) {
      std::vector<std::pair<int, int>> surfaceEntities;
      gmsh::model::getBoundary({{3, volumeTag}}, surfaceEntities, false, false,
                               false);
      std::pair<int, int> surface = surfaceEntities[0];
      if (surface.first == 2) {
        ++surfaceTag;
        gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
        gmsh::model::setPhysicalName(2, surfaceTag,
                                     "surface_" + std::to_string(i + 1));
      }
    }
  }
  for (int i = 1; i < numLayers; ++i) {
    gmsh::model::occ::remove({{3, i + 1}});
  }
  gmsh::model::occ::synchronize();

  std::vector<double> facesList(numLayers);
  for (int i = 0; i < numLayers; ++i) facesList[i] = i + 1;

  gmsh::model::mesh::field::add("Distance", 1);
  gmsh::model::mesh::field::setNumbers(1, "FacesList", facesList);

  gmsh::model::mesh::field::add("Threshold", 2);
  gmsh::model::mesh::field::setNumber(2, "InField", 1);
  gmsh::model::mesh::field::setNumber(2, "SizeMin", meshSizeMin);
  gmsh::model::mesh::field::setNumber(2, "SizeMax", meshSizeMax);
  gmsh::model::mesh::field::setNumber(2, "DistMin", 0.0);
  double fac = 10.0;
  gmsh::model::mesh::field::setNumber(2, "DistMax", meshSizeMin * fac);
  gmsh::model::mesh::field::setAsBackgroundMesh(2);

  // Generate 3D mesh
  gmsh::option::setNumber(
      "Mesh.Algorithm3D",
      algorithm);  
  gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
  gmsh::model::mesh::generate(3);

  // Write
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::write(outputFileName + ".msh");

  gmsh::finalize();
}

std::vector<double> parseRadii(const std::string &radiiStr) {
  std::vector<double> radii;
  std::istringstream iss(radiiStr);
  std::string token;

  while (std::getline(iss, token, '-')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                token.end());
    radii.push_back(std::stod(token));
  }

  return radii;
}

int main(int argc, char **argv) {
  double meshSizeMin = 150e3;
  double meshSizeMax = 600e3;
  int algorithm = 1;
  int elementOrder = 2;
  std::string inputFileName = "data/prem.200.noiso";
  std::string outputFileName = "data/prem";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-i" && i + 1 < argc) {
      inputFileName = argv[++i];
    } else if (arg == "-s" && i + 1 < argc) {
      std::string meshSizeStr = argv[++i];
      auto meshSizes = parseRadii(meshSizeStr);
      if (meshSizes.size() == 2) {
        meshSizeMin = meshSizes[0];
        meshSizeMax = meshSizes[1];
      } else {
        std::cerr << "Error: mesh sizes should have two values.\n";
        return 1;
      }
    } else if (arg == "-o" && i + 1 < argc) {
      outputFileName = argv[++i];
    } else if (arg == "-order" && i + 1 < argc) {
      elementOrder = std::stoi(argv[++i]);
    } else if (arg == "-alg" && i + 1 < argc) {
      algorithm = std::stod(argv[++i]);
    }
  }

  double R = 6371e3;
  std::vector<double> radii = extractLayerBoundaries(inputFileName, R);
  meshSizeMin /= R;
  meshSizeMax /= R;

  std::cout << "Detected radii of " << radii.size() << " layers: ";
  for (const double r : radii) {
    std::cout << std::fixed << std::setprecision(8) << r << " ";
  }
  std::cout << std::fixed << std::setprecision(2) << "(The length scale is "
            << R << " meters.)" << std::endl;

  // Run the spherical layers creation function
  createConcentricSphericalLayers(radii, meshSizeMin,
                                  meshSizeMax, elementOrder, algorithm,
                                  outputFileName);

  return 0;
}
