// Poisson Solver for the offset-sphere benchmark
// example run: mpirun -np 8 bin/ex1 -m data/benchmark.msh -o 2 -mth 1 -deg 16 -lin
// 0
#include "common.hpp"
#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<real_t>;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int myid = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char *mesh_file = "data/benchmark.msh";
  int order = 2;
  int deg = 8;
  int method = 1;
  int linearised = 0;

  real_t G_dim = 6.6743e-11;
  real_t rho_dim = 5e3;

  real_t L_scale = 7e6;
  real_t RHO_scale = rho_dim;
  real_t T_scale = 1.0 / std::sqrt(G_dim * RHO_scale);

  real_t start_time, end_time, assembly_time, solver_time;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&deg, "-deg", "--deg", "spherical-harmonic cutoff.");
  args.AddOption(
      &method, "-mth", "--method",
      "Solution method: 0 = Neumann, 1 = DtN, 2 = multipole, 10 = Dirichlet.");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve reference (0) or linearised (1) problem.");
  args.AddOption(&rho_dim, "-sphere_density", "--sphere_density",
                 "Constant density of the offset sphere.");
  args.AddOption(&L_scale, "-L", "--L", "Length scale.");
  args.AddOption(&T_scale, "-T", "--T", "Time scale.");
  args.AddOption(&RHO_scale, "-RHO", "--RHO", "Density scale.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) args.PrintOptions(cout);

  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const real_t rho_nd = nd.ScaleDensity(rho_dim);
  real_t G = -4.0 * pi * G_dim * RHO_scale * T_scale * T_scale;
  ConstantCoefficient G_coeff(G);
  if (myid == 0) {
    cout << "RHS dimensionless number G = " << G << endl;
  }

  // Mesh
  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  const int dim = pmesh.Dimension();

  Array<int> dom_marker(pmesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;
  Vector c1 = MeshCentroid(&pmesh, dom_marker);

  auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;
  auto [found1, same1, r1] = SphericalBoundaryRadius(&pmesh, bdr_marker, c1);
  auto outer_bdr_marker = ExternalBoundaryMarker(&pmesh);

  H1_FECollection fec(order, dim);
  L2_FECollection dfec(order - 1, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);

  // Finite-element Space
  std::unique_ptr<ParFiniteElementSpace> dfes;
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  if (method == 2) {
    dfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &dfec);
  }

  std::unique_ptr<ParFiniteElementSpace> vfes;
  if (linearised == 1) {
    vfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &dfec, dim);
  }

  // Assemble the bilinear and linear forms
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  ConstantCoefficient eps(0.001);
  ParBilinearForm as(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  auto rho_coeff_1 = ConstantCoefficient(rho_nd);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff_1);
  ProductCoefficient rhs_coeff(G_coeff, rho_coeff);

  auto x = ParGridFunction(&fes);

  ParLinearForm b(&fes);

  auto uv = Vector(dim);
  uv = 1.0;

  std::unique_ptr<ParGridFunction> u;
  if (linearised == 1) {
    auto uCoeff_1 = VectorConstantCoefficient(uv);
    auto uCoeff = PWVectorCoefficient(dim);
    uCoeff.UpdateCoefficient(1, uCoeff_1);
    u = std::make_unique<ParGridFunction>(vfes.get());
    u->ProjectCoefficient(uCoeff);
  }

  if (linearised == 0) {
    b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b.Assemble();

    if (method == 1 && dim == 2) {
      x = 1.0;
      auto mass = b(x);
      auto l = ParLinearForm(&fes);
      auto one = ConstantCoefficient(1.0);
      l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), outer_bdr_marker);
      l.Assemble();
      auto length = l(x);
      b.Add(-mass / length, l);
    }
  } else {
    auto d = ParMixedBilinearForm(&fes, vfes.get());
    d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rhs_coeff));
    d.Assemble();
    d.MultTranspose(*u, b);
  }

  if (method == 2) {
    if (linearised == 0) {
      if (myid == 0) {
        start_time = MPI_Wtime();
      }
      auto c = PoissonMultipoleOperator(MPI_COMM_WORLD, dfes.get(), &fes, deg);
      c.Assemble();
      if (myid == 0) {
        end_time = MPI_Wtime();
        assembly_time = (end_time - start_time);
      }
      auto rhof = GridFunction(dfes.get());
      rhof.ProjectCoefficient(rhs_coeff);
      c.AddMult(rhof, b, -1);
    } else {
      if (myid == 0) {
        start_time = MPI_Wtime();
      }
      auto c = PoissonLinearisedMultipoleOperator(MPI_COMM_WORLD, vfes.get(),
                                                  &fes, rhs_coeff, deg);
      c.Assemble();
      if (myid == 0) {
        end_time = MPI_Wtime();
        assembly_time = (end_time - start_time);
      }
      c.AddMult(*u, b, -1);
    }
  }

  // Solve
  x = 0.0;
  Array<int> ess_tdof_list{};
  if (method == 10) fes.GetEssentialTrueDofs(outer_bdr_marker, ess_tdof_list);

  if (method == 0) {
    ParLinearForm l(&fes);
    ConstantCoefficient one(1.0);
    l.AddDomainIntegrator(new DomainLFIntegrator(one));
    l.Assemble();

    ParGridFunction z(&fes);
    z = 1.0;
    const real_t volume = l(z);
    const real_t fint = b(z);

    b.Add(-(fint / volume), l);
  }

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  HypreParMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = HypreBoomerAMG(As);

  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(5000);
  solver.SetPrintLevel(1);

  if (method == 1) {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, deg);
    c.Assemble();
    if (myid == 0) {
      end_time = MPI_Wtime();
      assembly_time = (end_time - start_time);
    }
    auto C = c.RAP();
    auto D = SumOperator(&A, 1.0, &C, 1.0, false, false);
    solver.SetOperator(D);
    solver.SetPreconditioner(P);

    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    if (dim == 2) {
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    } else {
      solver.Mult(B, X);
    }
    if (myid == 0) {
      end_time = MPI_Wtime();
      solver_time = (end_time - start_time);
    }
  } else {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    solver.SetOperator(A);
    solver.SetPreconditioner(P);
    if (method == 10) {
      solver.Mult(B, X);
    } else {
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    }
    if (myid == 0) {
      end_time = MPI_Wtime();
      solver_time = (end_time - start_time);
    }
  }

  if (myid == 0) {
    std::cout << "Assembly time: " << assembly_time << " s" << std::endl;
    std::cout << "Solver time: " << solver_time << " s" << std::endl;
  }

  a.RecoverFEMSolution(X, b, x);

  // Post-processing
  auto exact = UniformSphereSolution(dim, c1, r1);
  auto exact_coeff_0 =
      linearised == 0 ? exact.Coefficient() : exact.LinearisedCoefficient(uv);

  ConstantCoefficient scale_rhs(-rho_nd * G / 4.0 / pi);
  ProductCoefficient exact_coeff(scale_rhs, exact_coeff_0);

  GridFunctionCoefficient x_coef(&x);
  real_t x_c = EvalCoefficientAtPointGlobal(x_coef, pmesh, c1);
  real_t y_c = EvalCoefficientAtPointGlobal(exact_coeff, pmesh, c1);

  auto y = ParGridFunction(&fes);
  y.ProjectCoefficient(exact_coeff);
  x -= x_c;
  y -= y_c;

  auto dif = ParGridFunction(x);
  dif -= y;

  auto submesh = ParSubMesh::CreateFromDomain(pmesh, dom_marker);
  auto subfes = ParFiniteElementSpace(&submesh, &fec);
  auto subdif = ParGridFunction(&subfes);
  auto suby = ParGridFunction(&subfes);
  submesh.Transfer(dif, subdif);
  suby.ProjectCoefficient(exact_coeff);
  auto zero = ConstantCoefficient(0);
  auto error = subdif.ComputeL2Error(zero);
  auto norm = suby.ComputeL2Error(zero);
  error /= norm;

  if (myid == 0) {
    cout << "Global relative L2 error over the offset sphere: " << error
         << endl;
  }

  std::filesystem::create_directories("results");
  const std::string mesh_base =
      std::filesystem::path(mesh_file).stem().string();
  std::ostringstream tag;
  tag << mesh_base << "_" << method << "_" << deg;
  std::ostringstream rank_suffix;
  rank_suffix << "." << std::setfill('0') << std::setw(6) << myid;

  const std::string submesh_path =
      "results/mesh_" + tag.str() + ".mesh" + rank_suffix.str();
  const std::string subsol_path =
      "results/sol_" + tag.str() + ".gf" + rank_suffix.str();

  {
    std::ofstream mesh_ofs(submesh_path);
    mesh_ofs.precision(8);
    submesh.Print(mesh_ofs);
  }

  {
    std::ofstream sol_ofs(subsol_path);
    sol_ofs.precision(8);
    subdif.Save(sol_ofs);
  }

}
