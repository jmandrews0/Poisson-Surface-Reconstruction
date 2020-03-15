// 211BProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <igl/cotmatrix.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/dot.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/marching_cubes.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stack>


double smoothF(const Eigen::Vector3d& q, const Eigen::Vector3d& p, double W)
{
	Eigen::Matrix3d Cov;
	Cov <<	1/1.0, 0, 0,
			0, 1/1.0, 0,
			0, 0, 1/1.0;
	double E = 2.71828;
	double PI = 3.14159;
	Eigen::Vector3d diff = (q - p);
	//std::cout << "det: " << Cov.determinant() << std::endl;
	return std::pow(E, -0.5*diff.transpose()*Cov*diff) / (std::sqrt(2 * PI*Cov.determinant())); // std::pow(W,3);
}


// second derivative of gaussian smoothF function
double laplacian(const Eigen::Vector3d& q, const Eigen::Vector3d& p, double W)
{
	//Eigen::Vector3d sigma(1, 1, 1);
	double E = 2.71828;
	double PI = 3.14159;
	double var = 1.0;
	Eigen::Vector3d diff = (q - p)/(2*W); //* 8.0;
	return (diff.transpose()*diff - std::pow(var, 2)) / std::pow(var, 4) * std::pow(E, -0.5 / std::pow(var, 2)*diff.transpose()*diff); // std::pow(W, 3);
}

// first derivative of gaussian smoothF function
double deriv(double q, double p, double W)
{
	double E = 2.71828;
	double PI = 3.14159;
	double var = 1.0;
	double diff = (q - p) / W;
	return (diff / std::pow(var, 2)) * std::pow(E, -0.5*std::pow(diff / var, 2)); // std::pow(W, 3);
}


int findCell(const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::Vector3d& C)
{
	bool found = false;
	int cell = 0;
	while (!found)
	{
		//std::cout << "        cell = " << cell << std::endl;
		// is this node a child?
		bool leafN = true;
		for (int j = 0; j < 8; j++)
		{
			if (CH(cell, j) != -1) leafN = false;
		}
		if (leafN)
		{
			return cell;
		}
		//iterate deeper
		//std::cout << "        iterating deeper..." << std::endl;
		int index = 0;
		if (C(0) >= CN(cell, 0)) index += 1;
		if (C(1) >= CN(cell, 1)) index += 2;
		if (C(2) >= CN(cell, 2)) index += 4;
		cell = CH(cell, index);
	}
	return cell;
}


// get array of leaf node indices of octree
//leafNodes(O_CH, O_CN, O_L, L_CN);
void leafNodes(const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, Eigen::VectorXi& Leaves, Eigen::MatrixXd& Leaves_CN)
{
	std::vector<int> L;
	std::vector<double> L_CN;
	for (int i = 0; i < CH.rows(); i++)
	{
		bool leaf = true;
		for (int j = 0; j < 8; j++)
		{
			if (CH(i,j) != -1) leaf = false;
		}
		if (leaf)
		{
			L.push_back(i);
			L_CN.push_back(CN(i,0));
			L_CN.push_back(CN(i,1));
			L_CN.push_back(CN(i,2));
			// get k nearest "same depth" nodes

		}
	}
	Leaves = Eigen::Map<Eigen::VectorXi>(L.data(), L.size());
	Leaves_CN = Eigen::Map<Eigen::MatrixXd>(L_CN.data(), 3, L.size()).transpose();
}


void getNeighborCells(const Eigen::VectorXi& L, Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::VectorXd& W, Eigen::MatrixXi& Neighbors)
{
	Neighbors = Eigen::MatrixXi(L.rows(), 7);
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}

	for (int i = 0; i < L.size(); i++)
	{
		//std::cout << "i = " << i << " -----------------------------------" << std::endl;
		// find nearest neighbors for this cell
		Neighbors(i, 0) = i;
		int index = L(i);
		Eigen::Vector3d C = CN.row(index);
		std::vector<Eigen::Vector3d> searchCenters;
		Eigen::Vector3d X(W(index), 0, 0);
		Eigen::Vector3d Y(0, W(index), 0);
		Eigen::Vector3d Z(0, 0, W(index));
		searchCenters.push_back(C - X);
		searchCenters.push_back(C + X);
		searchCenters.push_back(C - Y);
		searchCenters.push_back(C + Y);
		searchCenters.push_back(C - Z);
		searchCenters.push_back(C + Z);
		// iterate through all search centers looking for match
		for (int j=0; j < searchCenters.size(); j++)
		{
			int cell = findCell(CH, CN, searchCenters.at(j));
			Neighbors(i, j + 1) = leaf.at(cell);
		}
	}
}


Eigen::Vector3d gradientField(	const Eigen::Vector3d& q, // point to interpolate
								const Eigen::MatrixXd& V, // all vertices
								const Eigen::MatrixXd& N, // all normals
								const std::vector<std::vector<int>>& PI, // indices of points in each octree cell
								const Eigen::MatrixXi& CH, // children for each octree cell
								const Eigen::MatrixXd& CN, // center of each octree cell
								const Eigen::VectorXd& W) // nearest neighbors of each octree cell
{
	// get octree cell containing p
	//int cell = findCell(CH, CN, q);
	//Eigen::VectorXi nearest = O_I.row(cell);
	// iterate through all sample points
	int pcount = 0;
	Eigen::Vector3d interpVector(0,0,0);
	/*
	for (int c = 0; c < nearest.size(); c++)
	{
		int index = nearest(c);
		std::vector<int> points = PI.at(index);
		for (int s = 0; s < points.size(); s++)
		{
			interpVector += N.row(points.at(s)) * smoothF(q, CN.row(index), W(index)) * W(index);
			pcount++;
		}
	}
	*/
	for (int s = 0; s < V.rows(); s++)
	{
		int index = findCell(CH, CN, V.row(s)); // get index of octree cell for sample s
		interpVector += N.row(s) * smoothF(q, CN.row(index), 1.0); //* W(index);
		pcount++;
	}

	return interpVector / pcount;
}

void generateL(int x_res, int y_res, int z_res, Eigen::SparseMatrix<double>& Lap)
{
	int size = x_res * y_res * z_res;
	Lap = Eigen::SparseMatrix<double>(size, size);

	//for (int r = 0; r < size; r++)
	//{
		//for (int c = 0; c < size; c++)
		//{
			//Lap(r, c) = laplacian(CN.row(L(r)), CN.row(L(c)), W(L(r)));
		//	Lap.insert(r, c) = 0;
		//}
	//}
	for (int r = 0; r < size; r++)
	{
		if (r%x_res == 0) {
			Lap.insert(r, r) = 1e9;
			continue;
		}
		/*
		if (r%x_res == x_res - 1) {
			Lap.insert(r, r) = 1e9;
			continue;
		}
		if (r % (x_res*y_res) < x_res) {
			Lap.insert(r, r) = 1e9;
			continue;
		}
		*/
		Lap.insert(r, r) = 6;
		// x neighbors
		if(r < size-1) Lap.insert(r, r + 1) = -1;
		if(r > 0) Lap.insert(r, r - 1) = -1;
		// y neighbors
		if(r < size-x_res) Lap.insert(r, r + x_res) = -1;
		if(r > x_res) Lap.insert(r, r - x_res) = -1;
		// z neighbors
		if(r < size- x_res * y_res) Lap.insert(r, r + x_res*y_res) = -1;
		if(r > x_res * y_res) Lap.insert(r, r - x_res*y_res) = -1;
	}
}


int level(double width, double total)
{
	return (int)std::log2(total / width);
}

double minWidth(const Eigen::VectorXd& W)
{
	double min = W(0);
	for (int i = 1; i < W.rows(); i++)
	{
		if (W(i) <= min) min = W(i);
	}
	return min;
}


void makeDenseOctree(	const Eigen::MatrixXi& CH,
						const Eigen::MatrixXd& CN,
						const Eigen::VectorXd& W,
						const Eigen::VectorXi& L,
						const Eigen::VectorXd& sol,
						Eigen::VectorXd& denseSol,
						Eigen::MatrixXd& denseCenters,
						int& x_res, int& y_res, int& z_res,
						std::map<int,int>& leafToDense)
{
	std::vector<double> dSol;
	std::vector<double> dCN;
	// get max depth
	int max = 0;
	for (int i = 0; i < L.size(); i++)
	{
		int lev = level(W(L(i)), W(0));
		if (lev > max) max = lev;
	}
	max = max-1; // JUST FOR TESTING

	// make map from octree space to leaf space
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}

	x_res = std::pow(2, max);
	y_res = std::pow(2, max);
	z_res = std::pow(2, max);
	double step = W(0) / (double)std::pow(2, max);
	Eigen::Vector3d X(1, 0, 0);
	Eigen::Vector3d Y(0, 1, 0);
	Eigen::Vector3d Z(0, 0, 1);
	Eigen::Vector3d start = X * step / 2.0 + Y * step / 2.0 + Z * step / 2.0 - X * W(0) / 2.0 - Y * W(0) / 2.0 - Z * W(0) / 2.0;
	int count = 0;
	for (int x = 0; x < x_res; x++)
	{
		for (int y = 0; y < y_res; y++)
		{
			for (int z = 0; z < z_res; z++)
			{
				// get value from sol
				Eigen::Vector3d C = x * X*step + y * Y*step + z * Z*step + start;
				int cell = findCell(CH, CN, C);
				leafToDense[leaf.at(cell)] = count;

				dSol.push_back(sol(leaf.at(cell)));
				dCN.push_back(C(0));
				dCN.push_back(C(1));
				dCN.push_back(C(2));
				count++;
			}
		}
	}

	denseSol = Eigen::Map<Eigen::VectorXd>(dSol.data(), dSol.size());
	denseCenters = Eigen::Map<Eigen::MatrixXd>(dCN.data(), 3, dSol.size()).transpose();
}


double calcIsovalue(const Eigen::MatrixXd& V, const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::VectorXi& L, const Eigen::VectorXd& sol, const std::map<int, int>& leafToDense)
{
	std::cout << "calcIsovalue" << std::endl;
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}

	int cell = 0;
	double sum = 0;
	int count = 0;
	for (int i = 0; i < V.rows(); i++)
	{
		cell = findCell(CH, CN, V.row(i));
		std::cout << "found cell " << cell << std::endl;
		int l = leaf.at(cell);
		std::cout << "got leaf for cell " << l << std::endl;
		if (leafToDense.find(l) != leafToDense.end()) {
			std::cout << "    dense has leaf" << l << std::endl;
			int index = leafToDense.at(l);
			std::cout << "    got index for leaf" << index << std::endl;
			sum += sol(index);
			count++;
		}
		//Eigen::Vector3d loc = CN.row(cell);
		// search for correct solution index
		//std::cout << "added to sum" << std::endl;
	}
	return sum / count;
}


void applyEdgeConstraints(Eigen::MatrixXd& MatL, const Eigen::VectorXi& L, const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, Eigen::MatrixXd& Edges)
{
	std::vector<double> edges;
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}
	// search for all 8 corner nodes

	bool leafN = false;
	int cell = 0;
	while (!leafN)
	{
		for (int j = 0; j < 8; j++)
		{
			if (CH(cell, j) == -1) leafN = true;
		}
		if (!leafN) cell = CH(cell, 0);
	}
	std::cout << "selected cell: " << CN.row(cell) << std::endl;

	for (int j = 0; j < L.rows(); j++)
	{
		int l = L(j);
		if (std::abs(CN(l, 0)) >= std::abs(CN(cell,0))) { MatL(j, j) = 1e9; 
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
		else if (std::abs(CN(l, 1)) >= std::abs(CN(cell, 1))) { MatL(j, j) = 1e9;
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
		else if (std::abs(CN(l, 2)) >= std::abs(CN(cell, 2))) { MatL(j, j) = 1e9;
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
	}
		/*
		if (i == 0)
		{
			for (int j = 0; j < CN.rows(); j++)
			{
				int l = leaf.at(cell);
				if (CN.row(j)[0] < CN.row(cell)[0]) { MatL(l, l) = 1e9; count++; 
					edges.push_back(CN(j,0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[1] < CN.row(cell)[1]) { MatL(l, l) = 1e9; count++; 
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[2] < CN.row(cell)[2]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
			}
		}
		else if (i == 7)
		{
			for (int j = 0; j < CN.rows(); j++)
			{
				int l = leaf.at(cell);
				if (CN.row(j)[0] > CN.row(cell)[0]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[1] > CN.row(cell)[1]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[2] > CN.row(cell)[2]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
			}
		}
		*/
	Edges = Eigen::Map<Eigen::MatrixXd>(edges.data(), 3, edges.size()/3).transpose();
}


void extraPoints(const Eigen::MatrixXd& V, Eigen::MatrixXd& Extra)
{
	Extra = Eigen::MatrixXd(V.rows()+6, 3);
	double minX = V(0, 0); 	double maxX = V(0, 0);
	double minY = V(0, 1);	double maxY = V(0, 1);
	double minZ = V(0, 2);	double maxZ = V(0, 2);
	for (int i = 0; i < V.rows(); i++)
	{
		Extra.row(i) = V.row(i);
		if (V(i, 0) > maxX) maxX = V(i, 0);
		if (V(i, 1) > maxY) maxY = V(i, 1);
		if (V(i, 2) > maxZ) maxZ = V(i, 2);
		if (V(i, 0) < minX) minX = V(i, 0);
		if (V(i, 1) < minY) minY = V(i, 1);
		if (V(i, 2) < minZ) minZ = V(i, 2);
	}
	double diffX = maxX - minX;
	double diffY = maxY - minY;
	double diffZ = maxZ - minZ; 
	Extra.row(V.rows()) = Eigen::Vector3d(maxX + diffX / 2, 0, 0);
	Extra.row(V.rows()+1) = Eigen::Vector3d(minX - diffX / 2, 0, 0);
	Extra.row(V.rows()+2) = Eigen::Vector3d(0, maxY + diffY / 2, 0);
	Extra.row(V.rows()+3) = Eigen::Vector3d(0, minY - diffY / 2, 0);
	Extra.row(V.rows()+4) = Eigen::Vector3d(0, 0, maxZ + diffZ / 2);
	Extra.row(V.rows()+5) = Eigen::Vector3d(0, 0, minZ - diffZ / 2);
}
//------------------------------------------------------------------------------------------------------

int main()
{
	// load sample points
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::readOBJ("curve.obj",V,F);

	// Build octree for k nearest neighbors S_I
	std::vector<std::vector<int > > O_PI;
	Eigen::MatrixXi O_CH;
	Eigen::MatrixXd O_CN;
	Eigen::VectorXd O_W;
	igl::octree(V, O_PI, O_CH, O_CN, O_W);
	// get list of leaf nodes only
	Eigen::VectorXi O_L;
	Eigen::MatrixXd O_LCN;
	leafNodes(O_CH, O_CN, O_L, O_LCN);
	// find k nearest neighbors for sample points
	Eigen::MatrixXi S_I;
	int k = 8;
	igl::knn(V, k, O_PI, O_CH, O_CN, O_W, S_I);
	std::cout << "found k nearest neighbors" << std::endl;
	
	// Build octree for sample points
	std::vector<std::vector<int > > S_PI;
	Eigen::MatrixXi S_CH;
	Eigen::MatrixXd S_CN;
	Eigen::VectorXd S_W;
	Eigen::MatrixXd Extra;
	extraPoints(V, Extra);
	igl::octree(Extra, S_PI, S_CH, S_CN, S_W);
	std::cout << "second octree" << std::endl;
	// get list of leaf nodes only
	Eigen::VectorXi S_L;
	Eigen::MatrixXd S_LCN;
	leafNodes(S_CH, S_CN, S_L, S_LCN);
	
	// CALCULATE NORMALS FOR EACH POINT
	Eigen::MatrixXd N(V.rows(), 3); // normals to be calculated
	Eigen::MatrixXd M(k, 3); // covariance matrix to be calculated
	
	for (int r = 0; r < V.rows(); r++)
	{
		// calculate mean of points
		Eigen::Vector3d mean(0, 0, 0);
		for (int v = 0; v < S_I.cols(); v++)
		{
			mean += V.row(S_I(r, v));
		}
		mean /= S_I.cols();
		
		// calculate matrix M
		Eigen::Vector3d diff(0, 0, 0);
		for (int v = 0; v < S_I.cols(); v++)
		{
			diff = V.row(S_I(r, v)) - mean.transpose();
			M.row(v) = diff;
		}

		// SVD decomposition
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();
		Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k); rhs(0) = 1; // will have to be more dimensions for larger values of k
		N.row(r) = svd.solve(rhs).normalized();
	}
	std::cout << "calculated normals" << std::endl;

	// GENERATE VECTOR FIELD
	Eigen::MatrixXd S_N(S_LCN.rows(), 3);
	//int count = 0;
	//double min = minWidth(S_W);
	for (int i=0; i<S_LCN.rows(); i++)
	{
		//S_N.row(i) = Eigen::Vector3d(0, 0, 0);
		S_N.row(i) = gradientField(S_LCN.row(i), V, N, S_PI, S_CH, S_CN, S_W);
	}
	/*
	std::map<int, int> leaf;
	for (int i = 0; i < S_L.size(); i++)
	{
		leaf[S_L(i)] = i;
	}
	for (int i = 0; i < V.rows(); i++)
	{
		int cell = findCell(S_CH, S_CN, V.row(i));
		S_N.row(leaf.at(cell)) = N.row(i);
	}
	*/
	std::cout << "Generated Vector Field" << std::endl;

	// GET NEIGHBOR CELLS TO EACH CELL
	//Eigen::MatrixXi Neighbors;
	//getNeighborCells(S_L, S_CH, S_CN, S_W, Neighbors);
	//std::cout << "Found leaf cell Neighbors" << std::endl;

	// REFORMAT OCTREE SPACE TO HAVE CONSTANT RESOLUTION
	Eigen::VectorXd VN(S_N.rows());
	for (int i = 0; i < S_N.rows(); i++)
	{
		VN(i) = S_N.row(i).norm();
	}
	Eigen::VectorXd denseField;
	Eigen::MatrixXd denseCenters;
	int x_res; int y_res; int z_res;
	std::map<int, int> leafToDense;
	makeDenseOctree(S_CH, S_CN, S_W, S_L, VN, denseField, denseCenters, x_res, y_res, z_res, leafToDense);
	std::cout << "sol: " << S_N.size() << "denseSol: " << denseField.size() << "denseCenters" << denseCenters.size() << std::endl;
	std::cout << x_res << ", " << y_res << ", " << z_res << std::endl;
	
	// CREATE LAPLACIAN MATRIX
	Eigen::SparseMatrix<double> MatL; // make SparseMatrix<double>
	generateL(x_res, y_res, z_res, MatL);
	//Eigen::MatrixXd Edges;
	//applyEdgeConstraints(MatL, S_L, S_CH, S_CN, Edges);
	//std::cout << "Edge cells: " << Edges.rows() << std::endl;
	//MatL += Eigen::MatrixXd::Identity(MatL.rows(), MatL.cols())*1e-8;
	std::cout << "Generated Laplacian" << std::endl;

	// SOLVE min||Lx-v||^2 USING SVD
	//Eigen::VectorXd sol = MatL.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(VN);
	
	//Eigen::BDCSVD<Eigen::MatrixXd> bdcsvd(MatL, Eigen::ComputeThinU | Eigen::ComputeThinV);
	//Eigen::VectorXd sol = bdcsvd.solve(VN);
	//Eigen::VectorXd sol = (MatL.transpose() * MatL).ldlt().solve(MatL.transpose() * denseField);
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> lscg;
	lscg.compute(MatL);
	Eigen::VectorXd sol = lscg.solve(denseField);
	//sol = Eigen::VectorXd::Ones(sol.size()).normalized() - sol.normalized();
	std::cout << "Computed Indicator Function" << std::endl;

	// APPROXIMATE ISOVALUE NEAR SURFACE
	double isovalue = calcIsovalue(V, S_CH, S_CN, S_L, sol, leafToDense);
	std::cout << "isovalue = " << isovalue << std::endl;

	// MARCHING CUBES ALGORITHM
	Eigen::MatrixXd V_Recon;
	Eigen::MatrixXi F_Recon;
	igl::copyleft::marching_cubes(sol, denseCenters, x_res, y_res, z_res, isovalue, V_Recon, F_Recon);
	
	// PLOTTING DATA AND MESH
	const Eigen::RowVector3d green(0.2, 0.9, 0.2), blue(0.2, 0.2, 0.8), red(0.8, 0.2, 0.2), white(0.9, 0.9, 0.9), yellow(0.9,0.9,0.2);
	igl::opengl::glfw::Viewer viewer;
	/*
	for (int j = 0; j < S_L.size(); j++)
	{
		int i = S_L(j);
		Eigen::MatrixXd OctreeV1(8, 3);
		Eigen::MatrixXd OctreeV2(8, 3);
		Eigen::Vector3d X(S_W(i), 0.0, 0.0);
		Eigen::Vector3d Y(0.0, S_W(i), 0.0);
		Eigen::Vector3d Z(0.0, 0.0, S_W(i));
		OctreeV1 << (S_CN.row(i).transpose() - X / 2.0 - Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 - Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 - Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 - Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose();

		OctreeV2 << (S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose();
		viewer.data().add_edges(OctreeV1, OctreeV2, white);
	}
	*/
	Eigen::MatrixXd intensity(sol.rows(), 3);
	for (int i = 0; i < sol.rows(); i++)
	{
		intensity.row(i) = Eigen::Vector3d(0,sol(i),0);
	}
	
	std::ofstream myfile;
	myfile.open("sol.txt");
	myfile << sol;
	myfile.close();

	// Plot the mesh
	viewer.data().set_mesh(V_Recon, F_Recon);
	viewer.data().set_face_based(true);
	viewer.data().add_points(V, red);
	//viewer.data().add_edges(S_LCN + S_N, S_LCN, green);
	viewer.data().add_edges(V + N, V, green);
	//std::cout << "solution space: "<< denseSol.rows() << ", centers: " << denseCenters.rows() << ", intensity: " << intensity.size() << std::endl;
	//viewer.data().add_edges(denseCenters + intensity, denseCenters, white);
	viewer.launch();

	return 0;
}
